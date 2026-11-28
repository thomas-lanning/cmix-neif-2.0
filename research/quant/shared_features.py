"""Distill sixteen MLPs into four shared input banks with optional private features."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from allocation import bf16, round_bf16, np2
from distill import raster, hidden
from evaluate import BASELINE, REFERENCE_SHA, measure

FIT_ROWS = 12288
REFIT_LIMIT = 7
GROUP = 4


def phase(destination, stage):
    p = destination / "phase.tmp"
    p.write_text(stage + "\n")
    p.replace(destination / "status")
    print(stage, flush=True)


def save(destination, name, value):
    p = destination / (name + ".tmp")
    p.write_text(json.dumps(value, indent=2) + "\n")
    p.replace(destination / name)


def quantize_rows(weights):
    a = np.abs(weights).max(1) / REFIT_LIMIT
    a = np.maximum(a, 1e-10)
    for _ in range(4):
        codes = np.clip(np.rint(weights / a[:, None]), -REFIT_LIMIT, REFIT_LIMIT)
        a = np.maximum(
            1e-10, (weights * codes).sum(1) / np.maximum(1, (codes * codes).sum(1))
        )
    scales = round_bf16(a)
    codes = np.clip(np.rint(weights / bf16(scales)[:, None]), -REFIT_LIMIT, REFIT_LIMIT)
    return codes.astype(np.int8), scales


def key(layer, name):
    return f"blocks.{layer}.mlp.{name}"


def run(args):
    destination = args.output
    destination.mkdir(parents=True)
    profile = args.profile
    native = args.replay
    suite = args.suite
    phase(destination, "reading released model and native training receipts")
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == REFERENCE_SHA
    manifest = json.loads((profile / "manifest.json").read_bytes())
    assert manifest["model_sha256"] == REFERENCE_SHA
    training = set(manifest["documents"])
    initial = json.loads((suite / "manifest.json").read_bytes())
    search_ids = {w["document"] for w in initial["windows"]}
    assert not training & search_ids
    bank = np2.read_container(str(BASELINE))
    lookup = {v.name: v for v in bank}

    def val(l, n):
        return lookup[key(l, n)].a

    def act(l, d):
        return bf16(val(l, d + ".quantize_activation.scale")).item()

    def scale(l, d):
        return bf16(val(l, d + ".weight.scale")).reshape(-1)

    def q(l, d):
        return val(l, d + ".weight.q")

    candidate = [np2.Ent(t.name, t.code, t.a) for t in bank]
    out = {t.name: t for t in candidate}

    def assign(layer, n, array):
        out[key(layer, n)].a[:] = array.reshape(out[key(layer, n)].a.shape)

    shared_width = args.shared_width
    private_width = 1024 - shared_width
    fit_metrics = []
    for first in range(0, 16, GROUP):
        phase(
            destination,
            f"training shared feature bank for layers {first} through {first+3}",
        )
        uq = []
        us = []
        layers = []
        for layer in range(first, first + GROUP):
            xr = raster(profile, manifest, layer, "up")
            hr = raster(profile, manifest, layer, "down")
            assert len(xr) == len(hr) and q(layer, "up").shape == (1024, 192)
            gen = np.random.default_rng(21647 + layer)
            split = len(xr) * 7 // 8
            ids = np.r_[
                gen.choice(split, FIT_ROWS, replace=False),
                gen.choice(np.arange(split, len(xr)), 1024, replace=False),
            ]
            x = xr[ids].astype(np.float32) - 128
            h = hr[ids].astype(np.float32)
            up = q(layer, "up")
            down = q(layer, "down")
            _, _, check = hidden(
                x[-128:],
                up.astype(np.float32),
                scale(layer, "up") * act(layer, "up"),
                act(layer, "down"),
            )
            assert np.array_equal(
                check, h[-128:]
            ), f"native teacher mismatch layer {layer}"
            effective = down.astype(np.float32) * scale(layer, "down")[:, None]
            strength = h[:FIT_ROWS].var(0) * (effective**2).sum(0)
            ranking = np.argsort(-strength, kind="stable")
            private = ranking[:private_width]
            chosen = ranking[private_width : private_width + shared_width // GROUP]
            uq.append(up[chosen])
            us.append(val(layer, "up.weight.scale").reshape(-1)[chosen])
            layers.append((layer, x, (h @ effective.T) * act(layer, "down"), private))
        shared = np.concatenate(uq)
        shared_scale = np.concatenate(us)
        # Donor units retain native incoming weights and scales. Each destination
        # has its own activation scales and learns all its outgoing weights.
        assert len({r.tobytes() for r in shared}) == len(shared)
        for layer, x, y, private in layers:
            inputs = np.concatenate([shared, q(layer, "up")[private]])
            input_scales = np.concatenate(
                [shared_scale, val(layer, "up.weight.scale").reshape(-1)[private]]
            )
            assert len({r.tobytes() for r in inputs}) == len(inputs)
            ad = act(layer, "down")
            _, _, z = hidden(
                x, inputs.astype(np.float32), bf16(input_scales) * act(layer, "up"), ad
            )
            z *= ad
            a = z[:FIT_ROWS].astype(np.float64)
            b = y[:FIT_ROWS].astype(np.float64)
            gram = a.T @ a
            gram.flat[:: len(gram) + 1] += max(1e-9, np.trace(gram) / len(gram) * 0.003)
            weights = np.linalg.solve(gram, a.T @ b).T.astype(np.float32)
            codes, scales = quantize_rows(weights)
            reconstructed = codes.astype(np.float32) * bf16(scales)[:, None]
            loss = float(
                np.mean((z[FIT_ROWS:] @ reconstructed.T - y[FIT_ROWS:]) ** 2)
                / max(1e-12, np.mean(y[FIT_ROWS:] ** 2))
            )
            assert np.isfinite(loss)
            assign(layer, "up.weight.q", inputs)
            assign(layer, "up.weight.scale", input_scales)
            assign(layer, "down.weight.q", codes)
            assign(layer, "down.weight.scale", scales)
            fit_metrics.append(dict(layer=layer, heldout_block_relative_mse=loss))
        save(destination, "training.json", fit_metrics)
    phase(destination, "packing shared model and verifying lossless weight expansion")
    stored = [np2.Ent(t.name, t.code, t.a) for t in candidate]
    for t in stored:
        parts = t.name.split(".")
        if (
            len(parts) == 6
            and parts[0] == "blocks"
            and parts[2:5] == ["mlp", "up", "weight"]
            and int(parts[1]) % GROUP
        ):
            assert parts[-1] in ("q", "scale")
            t.a.reshape(1024, -1)[:shared_width].fill(0)
    packed = destination / "shared-v5.c5"
    np2.wc.write_tensor_file_v5(str(packed), np2.entries_of(stored), k=True)
    decoded = np2.wc.read_tensor_file_v3(str(packed))
    for l in range(16):
        if l % GROUP:
            for suffix in ["up.weight.q", "up.weight.scale"]:
                dtype, array = decoded[key(l, suffix)]
                owner = decoded[key(l - l % GROUP, suffix)][1]
                restored = array.copy()
                restored.reshape(1024, -1)[:shared_width] = owner.reshape(1024, -1)[
                    :shared_width
                ]
                decoded[key(l, suffix)] = dtype, restored
    for t in candidate:
        dtype, arr = decoded[t.name]
        assert t.same(np2.Ent(t.name, dtype, arr)), t.name
    table_bytes = packed.stat().st_size
    frozen = dict(
        format="v5 with shared up projection and scales expanded from first bank of each group of four; fixed group definition in code",
        shared_feature_banks=4,
        shared_rows_per_layer=shared_width,
        private_rows_per_layer=private_width,
        original_up_matrices=16,
        stored_bytes_each=table_bytes,
        both_copy_saving_against_release=2 * (BASELINE.stat().st_size - table_bytes),
        both_copy_saving_against_comparison=2 * (args.comparison_bytes - table_bytes),
        fixed_model_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),
        fully_roundtripped_expanded_tensors=True,
        training_documents=sorted(training),
        source="released predictor activations",
        training_document_count=len(training),
    )
    save(destination, "package.json", frozen)
    phase(
        destination,
        "exporting equivalent expanded model for native prediction benchmark",
    )
    payload, _, expected = np2.encode_np(
        candidate, True, np2.np_groups(candidate, True, True), True
    )
    expanded = destination / "expanded.c5"
    expanded.write_bytes(payload)
    check = np2.read_container(str(expanded))
    assert len(check) == len(expected) and all(
        a.same(b) for a, b in zip(check, expected)
    )
    phase(destination, "testing the full predictor on independent articles")
    screen = measure(native, suite, expanded, destination / "native", "shared-four")
    delta = 2 * (table_bytes - BASELINE.stat().st_size)

    def priced(result):
        return {
            k: result[k] - result["twice_weight_delta"] + delta
            for k in [
                "search_proxy_bytes",
                "confirmation_proxy_bytes",
                "proxy_delta_bytes",
            ]
        }

    priced_screen = priced(screen)
    save(
        destination,
        "screen.json",
        dict(
            expanded_sha256=screen["candidate_sha256"],
            stored_twice_delta=delta,
            predicted_full_score_delta=priced_screen,
            bits_per_target_delta=screen["bits_per_target_delta_by_window"],
            caveat="Native transformer replay. Excludes full cmix transfer and expansion code.",
        ),
    )
    if (
        max(
            priced_screen["search_proxy_bytes"],
            priced_screen["confirmation_proxy_bytes"],
        )
        < 0
    ):
        phase(destination, "testing frozen candidate on 96 further disjoint articles")
        large = args.holdout
        m = json.loads((large / "manifest.json").read_bytes())
        available = [
            w for w in m["windows"] if w["document"] not in training | search_ids
        ]
        count = min(96, len(available) // 2 * 2)
        assert count >= 12
        keep = [
            available[i] for i in np.linspace(0, len(available) - 1, count, dtype=int)
        ]
        hold = destination / "holdout"
        hold.mkdir()
        for w in keep:
            (hold / w["file"]).symlink_to(large / w["file"])
        m["windows"] = keep
        (hold / "manifest.json").write_text(json.dumps(m))
        validation = measure(
            native, hold, expanded, destination / "native", "shared-four-large"
        )
        save(
            destination,
            "validation.json",
            dict(documents=count, predicted_full_score_delta=priced(validation)),
        )
    phase(destination, "completed native screen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("replay", "profile", "suite", "holdout", "output"):
        parser.add_argument(label, type=lambda value: Path(value).resolve())
    parser.add_argument(
        "--comparison-bytes",
        type=int,
        default=BASELINE.stat().st_size,
        help="packed bytes per model copy for a second comparison",
    )
    parser.add_argument(
        "--shared-width",
        type=int,
        choices=(768, 1024),
        default=1024,
        help="shared rows out of 1024, with the remaining rows fitted separately per layer",
    )
    run(parser.parse_args())

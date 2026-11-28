"""Distill the released predictor into a causal dilated network with no teacher inputs at inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VOCAB = 208
WIDTH = 128
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
WARM = 2 * sum(DILATIONS)
SCORED = 192
FULL_TARGETS = 487298491


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mix = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.gate = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)
        self.left = 2 * dilation

    def forward(self, x):
        u = self.norm(x).transpose(1, 2)
        xmix = self.mix(F.pad(u, (self.left, 0))).transpose(1, 2)
        a, b = self.gate(xmix).chunk(2, dim=-1)
        return x + self.out(a * torch.sigmoid(b))


class Student(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.symbol = nn.Embedding(256, width)
        self.context = nn.ModuleList(nn.Embedding(256, 16) for _ in range(4))
        self.side = nn.Linear(64 + VOCAB, width)
        self.blocks = nn.ModuleList(Block(width, d) for d in DILATIONS)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, VOCAB)
        self.prior_weight = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.prior_weight.weight)
        nn.init.zeros_(self.prior_weight.bias)

    def forward(self, tokens, priors):
        base = priors.float().clamp_min(1e-8).log()
        centered = base - base.mean(dim=-1, keepdim=True)
        aux = [embed(tokens[..., i + 1]) for i, embed in enumerate(self.context)]
        x = self.symbol(tokens[..., 0]) + self.side(
            torch.cat([*aux, centered * 0.1], dim=-1)
        )
        for block in self.blocks:
            x = block(x)
        h = self.norm(x)
        return (
            self.output(h)
            + (0.25 + 1.5 * torch.sigmoid(self.prior_weight(h))) * centered
        )


def restore(blob):
    model = Student()
    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            shape = tuple(parameter.shape)
            if parameter.ndim <= 1:
                n = parameter.numel()
                recovered = np.frombuffer(
                    blob, dtype="<f2", count=n, offset=cursor
                ).astype(np.float32)
                cursor += 2 * n
            else:
                rows = shape[0]
                size = parameter.numel()
                scales = np.frombuffer(blob, "<f4", count=rows, offset=cursor)
                cursor += 4 * rows
                codes = np.frombuffer(blob, "i1", count=size, offset=cursor).reshape(
                    rows, -1
                )
                cursor += size
                recovered = codes * scales[:, None]
            parameter.copy_(torch.from_numpy(recovered.reshape(shape)))
    assert cursor == len(blob)
    return model.eval()


def export(net):
    """One signed byte per weight and an fp32 scale per output row. Biases use fp16."""
    packed = bytearray()
    for param in net.parameters():
        v = param.detach().cpu().numpy()
        if v.ndim <= 1:
            packed.extend(v.astype("<f2").tobytes())
        else:
            flat = v.reshape(v.shape[0], -1)
            scales = np.maximum(1e-10, np.abs(flat).max(1) / 127).astype("<f4")
            codes = np.rint(flat / scales[:, None]).clip(-127, 127).astype("i1")
            packed.extend(scales.tobytes())
            packed.extend(codes.tobytes())
    blob = bytes(packed)
    return blob, restore(blob)


def array_data(suite, w, profile):
    raw = (suite / w["file"]).read_bytes()
    assert raw[:8] == b"NEIFQ001" and hashlib.sha256(raw).hexdigest() == w["sha256"]
    arr = np.frombuffer(raw, offset=8, dtype=np.uint8).reshape(-1, 424)
    assert len(arr) == w["rows"] and arr[0, 7] == 1 and arr[1:, 7].sum() == 0
    scored = arr[:, 6].astype(bool)
    truth = arr[:, 5].copy()
    prob = (
        np.frombuffer(raw, offset=8, dtype=np.uint8)
        .reshape(-1, 424)[:, 8:]
        .copy()
        .view("<f2")
    )
    p = np.memmap(profile, dtype="<f4", mode="r").reshape(-1, 400)[:, 192:]
    assert (
        len(p) == int(scored.sum()) == w["targets"]
        and np.isfinite(p).all()
        and np.abs(p.sum(1) - 1).max() < 0.002
    )
    teacher = np.zeros((len(arr), VOCAB), dtype="<f2")
    teacher[scored] = p
    tolerance = abs(
        np.log2(p[np.arange(len(p)), truth[scored]].clip(3e-8)).sum()
        - np.log2(teacher[scored, truth[scored]].astype(np.float32).clip(3e-8)).sum()
    ) / len(p)
    assert tolerance < 0.001
    return arr[:, :5].copy(), prob, truth, scored, teacher


def crop(doc, start):
    # Advance only causal context. Positions before the article reset are masked out.
    stop = min(len(doc[0]), start + SCORED)
    lo = max(0, start - WARM)
    left = WARM - (start - lo)
    right = SCORED - (stop - start)
    batch = []
    for a in doc:
        pad = [(left, right)] + [(0, 0)] * (a.ndim - 1)
        batch.append(np.pad(a[lo:stop], pad))
    batch[3][:WARM] = False
    return batch


def tensors(crops, device):
    return [
        torch.as_tensor(np.stack(a), device=device, dtype=dtype)
        for a, dtype in zip(
            zip(*crops),
            [torch.long, torch.float32, torch.long, torch.bool, torch.float32],
            strict=True,
        )
    ]


@torch.no_grad()
def compare(model, docs):
    target_bits = teacher_bits = 0.0
    count = 0
    model.eval()
    for doc in docs:
        seq = [crop(doc, start) for start in range(256, len(doc[0]), SCORED)]
        for off in range(0, len(seq), 4):
            tok, prior, y, mask, teacher = tensors(seq[off : off + 4], "cpu")
            logits = model(tok, prior)
            lp = F.log_softmax(logits[mask], dim=-1)
            y = y[mask]
            if len(y) == 0:
                continue
            target_bits -= lp.gather(1, y[:, None]).sum().item() / math.log(2)
            ref = teacher[mask]
            ref = ref / ref.sum(1, keepdim=True).clamp_min(1e-10)
            teacher_bits -= (
                ref.gather(1, y[:, None]).clamp_min(3e-8).log2().sum().item()
            )
            count += len(y)
    assert count == sum(int(d[3].sum()) for d in docs)
    return dict(
        targets=count,
        student_bits=target_bits,
        teacher_bits=teacher_bits,
        extra_bits_per_target=(target_bits - teacher_bits) / count,
    )


def smoke():
    torch.set_num_threads(2)
    torch.manual_seed(1337)
    m = Student()
    tokens = torch.randint(256, (2, 400, 5))
    p = torch.rand(2, 400, 208)
    p /= p.sum(-1, keepdim=True)
    target = torch.randint(VOCAB, (2, 400))
    opt = torch.optim.AdamW(m.parameters(), lr=0.001)
    for _ in range(2):
        opt.zero_grad()
        loss = F.cross_entropy(m(tokens, p).flatten(0, 1), target.flatten())
        loss.backward()
        opt.step()
    blob, m = export(m)
    with torch.no_grad():
        original = m(tokens, p)
        other = tokens.clone()
        other[:, 201:] = 0
        assert torch.equal(original[:, :201], m(other, p)[:, :201])
    print(
        json.dumps(
            dict(
                weight_bytes=len(blob),
                parameters=sum(a.numel() for a in m.parameters()),
                causal_prefix_test=True,
            )
        )
    )


def main(args):
    from profile_cache import ProfileCache

    torch.set_num_threads(1)
    torch.manual_seed(20261128)
    d = args.output
    d.mkdir()

    def put(name, data):
        path = d / (name + ".tmp")
        path.write_text(json.dumps(data, indent=2) + "\n")
        path.replace(d / name)

    def stage(s):
        (d / "status").write_text(s + "\n")
        print(s, flush=True)

    model = Student().to(args.device)
    cache = ProfileCache(args.cache, args.profiler, args.teacher)
    windows = json.loads((args.suite / "manifest.json").read_text())["windows"]
    assert len(windows) == 1024 and len({w["document"] for w in windows}) == 1024
    stage("preparing verified native teacher distributions on 1024 articles")
    docs = []
    for i, w in enumerate(windows):
        path, receipt, _ = cache.get(args.suite, w)
        docs.append(array_data(args.suite, w, path))
        if (i + 1) % 128 == 0:
            print("loaded", i + 1, flush=True)
    fit = [doc for i, doc in enumerate(docs) if i % 8 < 6]
    tune = [doc for i, doc in enumerate(docs) if i % 8 == 6]
    test = [doc for i, doc in enumerate(docs) if i % 8 == 7]
    opt = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.01)
    rng = np.random.default_rng(558103)
    best_loss = float("inf")
    history = []
    for epoch in range(1, 9):
        stage(f"training all network layers, epoch {epoch} of 8")
        order = rng.permutation(len(fit))
        model.train()
        for off in range(0, len(order), 4):
            crops = []
            for i in order[off : off + 4]:
                doc = fit[i]
                where = np.flatnonzero(doc[3])
                center = int(rng.choice(where))
                start = max(256, min(center - SCORED // 2, len(doc[0]) - SCORED))
                crops.append(crop(doc, start))
            tok, prior, y, mask, teacher = tensors(crops, args.device)
            if not mask.any():
                continue
            out = model(tok, prior)[mask]
            lp = F.log_softmax(out, dim=-1)
            soft = teacher[mask]
            soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-10)
            loss = (
                -0.5 * (soft * lp).sum(-1).mean()
                - 0.5 * lp.gather(1, y[mask][:, None]).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        # Tune on predeclared documents and freeze the raw fixed-width package.
        blob, frozen = export(model)
        assessment = compare(frozen, tune[::8])
        history.append(dict(epoch=epoch, bytes_each=len(blob), **assessment))
        put("training.json", history)
        if assessment["student_bits"] < best_loss:
            best_loss = assessment["student_bits"]
            (d / "frozen.bin").write_bytes(blob)

    accepted = restore((d / "frozen.bin").read_bytes())
    put(
        "other-selection.json",
        compare(accepted, [a for i, a in enumerate(tune) if i % 8]),
    )
    stage("frozen independent evaluation on 128 unseen articles")
    result = compare(accepted, test)
    n = (d / "frozen.bin").stat().st_size
    result.update(
        fit_articles=len(fit),
        selection_articles=len(tune[::8]),
        test_articles=len(test),
        packed_weight_bytes_per_copy=n,
        native_teacher_copy_bytes=3744629,
        two_copy_saving_against_current=2 * (3717390 - n),
        predicted_package_delta_vs_release=2 * (n - 3744629)
        + result["extra_bits_per_target"] * FULL_TARGETS / 8,
        scope="Independent native transformer-output comparison. Cmix mixer/cache interaction, new runtime code and end-to-end archive are unmeasured.",
    )
    put("result.json", result)
    stage("completed independent screen")


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        smoke()
    else:
        p = argparse.ArgumentParser(description=__doc__)
        for n in ["suite", "cache", "profiler", "teacher", "output"]:
            p.add_argument(n, type=lambda s: Path(s).resolve())
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        main(p.parse_args())

"""Build or collect MLP activation profiles on discovery documents."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def instrument(root):
    file = root / "src/tf1/opt/model_opt.cpp"
    src = file.read_text()
    anchor = "      PSCOPE(PS_MM_MLP_DOWN);"
    if src.count(anchor) != 1:
        raise ValueError("release MLP changed")
    # down activations are dense uint8 codes in [0,127], immediately before
    # multiplication by the released down projection. q5 uses the native
    # unsigned representation of signed, biased input to the up projection.
    call = """
      for (int j = 0; j < n; ++j) {
        capture_codes(l, false, r[j].q768, mm.down.d_in);
        capture_codes(l, true, r[j].q5, D);
      }"""
    prefix = r'''
#include <cstdio>
#include <cstdlib>
static void capture_codes(int layer, bool up, const uint8_t* codes, int width) {
  static FILE* files[2][16]{};
  FILE*& file = files[up][layer];
  if (!file) {
    char path[80];
    std::snprintf(path, sizeof(path), "layer-%02d-%s.u8", layer, up ? "up" : "down");
    file = std::fopen(path, "ab");
  }
  if (!file || std::fwrite(codes, 1, width, file) != size_t(width)) std::exit(5);
}
'''
    file.write_text("#include <cstdint>\n" + prefix + src.replace(anchor, anchor + call))


def collect(replay, suite, model, output):
    from evaluate import model_id, REFERENCE_SHA
    suite_hash = hashlib.sha256((suite / "manifest.json").read_bytes()).hexdigest()
    rows = json.loads((suite / "manifest.json").read_bytes())["windows"][::2]
    sha, fnv, _ = model_id(model)
    if sha != REFERENCE_SHA:
        raise ValueError("start activation calibration from the released model")
    output.mkdir()
    for w in rows:
        win = suite / w["file"]
        if hashlib.sha256(win.read_bytes()).hexdigest() != w["sha256"]:
            raise ValueError("window hash differs")
        result = json.loads(subprocess.check_output([replay, model, fnv, win], cwd=output, text=True, timeout=300))
        if result["targets"] != w["targets"]:
            raise ValueError("scored targets differ")
    files = []
    for layer in range(16):
        for family, width in (("up", 192), ("down", 1024)):
            file = output / f"layer-{layer:02d}-{family}.u8"
            size = file.stat().st_size
            if not size or size % width:
                raise ValueError(f"invalid activation raster {file}")
            files.append(dict(file=file.name, rows=size//width, width=width,
                              sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (output / "manifest.json").write_text(json.dumps(dict(suite_sha256=suite_hash,
        model_sha256=sha, replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
        documents=[w["document"] for w in rows], files=files), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("instrument").add_argument("root", type=Path)
    p = sub.add_parser("collect")
    for name in ("replay","suite","model","output"):
        p.add_argument(name, type=Path)
    a = parser.parse_args()
    if a.action == "instrument":
        instrument(a.root)
    else:
        collect(*(p.resolve() for p in (a.replay,a.suite,a.model,a.output)))

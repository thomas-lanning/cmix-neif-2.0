"""Cache exact native head profiles for independent, document-reset fixtures on Linux."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from evaluate import model_id


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ProfileCache:
    def __init__(self, cache, executable, model):
        self.exe = executable.resolve()
        self.model = model.resolve()
        model_sha, self.fnv, _ = model_id(self.model)
        self.directory = cache / "native-head-f32-v1" / file_sha(self.exe) / model_sha
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, suite, window):
        fixture = suite / window["file"]
        key = window["sha256"]
        if fixture.parent != suite or file_sha(fixture) != key:
            raise ValueError(f"fixture differs from manifest: {fixture}")
        data = self.directory / (key + ".f32")
        info = self.directory / (key + ".json")
        expected_size = window["targets"] * 400 * 4
        with (self.directory / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if info.exists() and data.exists() and data.stat().st_size == expected_size:
                try:
                    receipt = json.loads(info.read_bytes())
                except (ValueError, UnicodeError):
                    receipt = {}
                if (
                    isinstance(receipt, dict)
                    and receipt.get("rows") == window["rows"]
                    and receipt.get("targets") == window["targets"]
                    and isinstance(receipt.get("bits"), (int, float))
                    and math.isfinite(receipt["bits"])
                    and file_sha(data) == receipt.get("sha256")
                ):
                    return data, receipt, True
            with tempfile.TemporaryDirectory(dir=self.directory) as folder:
                temporary = Path(folder) / "output"
                raw = subprocess.check_output(
                    [self.exe, self.model, self.fnv, fixture, temporary], timeout=300, text=True
                )
                receipt = json.loads(raw)
                if (
                    receipt["rows"] != window["rows"]
                    or receipt["targets"] != window["targets"]
                    or not math.isfinite(receipt["bits"])
                    or temporary.stat().st_size != expected_size
                ):
                    raise ValueError(f"incomplete native head profile for {fixture}")
                receipt["sha256"] = file_sha(temporary)
                marker = Path(folder) / "receipt"
                marker.write_text(json.dumps(receipt))
                os.replace(temporary, data)
                os.replace(marker, info)
            return data, receipt, False

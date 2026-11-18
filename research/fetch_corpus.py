"""Fetch enwik9 from the benchmark author or its public Kaggle mirror and verify it."""
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile


URLS = ("https://mattmahoney.net/dc/enwik9.zip",
        "https://api.kaggle.com/v1/datasets/download/jamesmcguigan/hutter-prize?datasetVersionNumber=1")
EXPECTED = "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"


def fetch(output):
    if output.exists():
        raise FileExistsError(output)
    for url in URLS:
        try:
            with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
                archive = Path(scratch)/"input.zip"
                with urllib.request.urlopen(url,timeout=90) as response,archive.open("wb") as f:
                    shutil.copyfileobj(response,f,1<<20)
                with zipfile.ZipFile(archive) as z:
                    member = z.getinfo("enwik9")
                    if member.file_size!=1_000_000_000:
                        raise ValueError("wrong uncompressed corpus length")
                    temporary = Path(scratch)/"enwik9"
                    sha=hashlib.sha256()
                    with z.open(member) as src,temporary.open("wb") as dst:
                        while chunk:=src.read(1<<20):
                            sha.update(chunk)
                            dst.write(chunk)
                if sha.hexdigest()!=EXPECTED:
                    raise ValueError("uncompressed corpus checksum differs")
                os.link(temporary,output)
            print("Verified enwik9",EXPECTED,"from",url,flush=True)
            return
        except (OSError,EOFError,ValueError,KeyError,zipfile.BadZipFile) as e:
            print("Corpus source failed",url,type(e).__name__,flush=True)
    raise RuntimeError("neither public source supplied the verified corpus")


if __name__=="__main__":
    fetch(Path(sys.argv[1]))

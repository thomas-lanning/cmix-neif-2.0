"""Assemble a decoder test using the published payload and a candidate's native components."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

REFERENCE="88dd7087404122d010261e6855afd88875d34dde1f0539377d3e832fc0f6b3fd"
WEIGHTS="729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assemble(archive, decoder, weights, destination):
    released=archive.read_bytes()
    if digest(released)!=REFERENCE:
        raise ValueError("published archive digest mismatch")
    payload_size,weights_size=struct.unpack_from("<II",released,len(released)-8)
    start=len(released)-8-payload_size
    stub_size=start-weights_size
    if stub_size!=183885 or weights_size!=3744629 or digest(released[stub_size:start])!=WEIGHTS:
        raise ValueError("unexpected published archive layout")
    image=decoder.read_bytes()
    model=weights.read_bytes()
    if image[:4]!=b"\x7fELF" or not model.startswith((b"FX2TFWP5",b"NEIFHYB2")):
        raise ValueError("unexpected decoder or weight file")
    payload=released[start:-8]
    with destination.open("xb") as f:
        for part in (image,model,payload,struct.pack("<II",len(payload),len(model))):
            f.write(part)
    return dict(origin="reassembled with the published payload",reference_archive_sha256=REFERENCE,
                decoder_bytes=len(image),decoder_sha256=digest(image),weight_bytes=len(model),
                weight_sha256=digest(model),payload_bytes=len(payload),payload_sha256=digest(payload),
                archive_bytes=destination.stat().st_size,archive_sha256=digest(destination.read_bytes()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("published_archive","candidate_decoder","candidate_weights","fresh_output"):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    print(json.dumps(assemble(a.published_archive,a.candidate_decoder,a.candidate_weights,a.fresh_output),indent=2))

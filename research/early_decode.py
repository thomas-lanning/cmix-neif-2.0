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


def split_compressor(compressor):
    """Mirror upstream decomp_prefix_len and the three-field compressor trailer."""
    comp=compressor.read_bytes()
    hints,order,model_size=struct.unpack_from("<III",comp,len(comp)-12)
    size=len(comp)-12-hints-order-model_size
    if size<=144 or model_size==0 or comp[:4]!=b"\x7fELF" or comp[56:58]!=b"\1\0":
        raise ValueError("unexpected compressor format")
    offset,length,_,image_size,cut,packed_cut=struct.unpack_from("<6I",comp,120)
    if not (offset and packed_cut and offset+length==size and packed_cut<=length and 0<cut<image_size):
        raise ValueError("unexpected packed code layout")
    decoder=bytearray(comp[:offset+packed_cut])
    struct.pack_into("<II",decoder,124,packed_cut,cut)
    model=comp[size+hints+order:-12]
    return bytes(decoder),model


def assemble(archive, image, model, destination):
    released=archive.read_bytes()
    if digest(released)!=REFERENCE:
        raise ValueError("published archive digest mismatch")
    payload_size,weights_size=struct.unpack_from("<II",released,len(released)-8)
    start=len(released)-8-payload_size
    stub_size=start-weights_size
    if stub_size!=183885 or weights_size!=3744629 or digest(released[stub_size:start])!=WEIGHTS:
        raise ValueError("unexpected published archive layout")
    if image[:4]!=b"\x7fELF" or not model.startswith((b"FX2TFWP5",b"NEIFHYB2",b"NEIFCHN1")):
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
    for n in ("published_archive","candidate_compressor","fresh_output"):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    decoder,model=split_compressor(a.candidate_compressor)
    evidence=assemble(a.published_archive,decoder,model,a.fresh_output)
    comp_size=a.candidate_compressor.stat().st_size
    evidence.update(comp_bytes=comp_size,comp_sha256=digest(a.candidate_compressor.read_bytes()),
                    entry_bytes_if_payload_identical=comp_size+evidence["archive_bytes"])
    print(json.dumps(evidence,indent=2))

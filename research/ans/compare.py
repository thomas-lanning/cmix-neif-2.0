"""Compare the upstream binary arithmetic coder with blockwise byte-renormalized rANS."""
import argparse
import json
import math
from pathlib import Path
import struct

EVENT = struct.Struct("<IB")  # Final model P(bit=1) in units of 2^-32, observed bit
MASK = (1 << 32) - 1
L = 1 << 40  # fits precision up to 32 and eight bits of renormalization in uint64


def arithmetic(events):
    low, high = 0, MASK
    payload = bytearray()
    for p, bit in events:
        mid = low + ((high-low)*p >> 32)
        if bit:
            high = mid
        else:
            low = mid + 1
        while low >> 24 == high >> 24:
            payload.append(high >> 24)
            low = (low << 8) & MASK
            high = ((high << 8) & MASK) | 255
    payload.append(high >> 24)
    return bytes(payload)


def frequency(p, precision):
    total = 1 << precision
    # Round to nearest while preserving nonzero mass for both symbols.
    return min(total-1, max(1, (p*total + (1 << 31)) >> 32))


def encode(block, precision):
    total = 1 << precision
    state, suffix = L, bytearray()
    for p, bit in reversed(block):
        f1 = frequency(p, precision)
        start, freq = (0, f1) if bit else (f1, total-f1)
        while state >= ((L >> precision) << 8)*freq:
            suffix.append(state & 255)
            state >>= 8
        state = ((state//freq) << precision) + state%freq + start
    return struct.pack("<Q", state) + suffix[::-1]


def decode(payload, probabilities, precision):
    total = 1 << precision
    state, = struct.unpack_from("<Q", payload)
    pos = 8
    bits = bytearray()
    for p in probabilities:
        f1 = frequency(p, precision)
        slot = state & (total-1)
        bit = slot < f1
        start, freq = (0, f1) if bit else (f1, total-f1)
        state = freq*(state >> precision) + slot-start
        while state < L:
            if pos == len(payload):
                raise ValueError("truncated rANS block")
            state = (state << 8) | payload[pos]
            pos += 1
        bits.append(bit)
    if state != L or pos != len(payload):
        raise ValueError("invalid terminal state or trailing bytes")
    return bits


def compare(events, precision, block_size):
    if not events or not 1 <= precision <= 32 or not 1 <= block_size <= (1 << 24):
        raise ValueError("invalid trace or coding parameters")
    stream = bytearray()
    for i in range(0, len(events), block_size):
        block = encode(events[i:i+block_size], precision)
        stream += struct.pack("<I", len(block)) + block
    cursor = 0
    for i in range(0, len(events), block_size):
        block = events[i:i+block_size]
        length, = struct.unpack_from("<I", stream, cursor)
        cursor += 4
        decoded = decode(stream[cursor:cursor+length], (p for p,_ in block), precision)
        if decoded != bytes(b for _,b in block):
            raise ValueError("rANS decoded different decisions")
        cursor += length
    if cursor != len(stream):
        raise ValueError("trailing frames")
    return len(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--precision", nargs="+", type=int, default=[12, 16, 24, 32])
    parser.add_argument("--block-size", nargs="+", type=int, default=[4096, 65536])
    parser.add_argument("--original-payload", type=Path)
    args = parser.parse_args()
    events = list(EVENT.iter_unpack(args.trace.read_bytes()))
    if any(not 0 < p <= MASK or b > 1 for p,b in events):
        raise ValueError("invalid model decision")
    reference = arithmetic(events)
    ideal_bytes = math.fsum(-math.log2((p if b else (1 << 32) - p) / (1 << 32))
                            for p, b in events) / 8
    if args.original_payload:
        # The last byte terminates the prefix in isolation. The release stream continues.
        with args.original_payload.open("rb") as f:
            f.seek(5)
            if f.read(len(reference)-1) != reference[:-1]:
                raise ValueError("arithmetic bytes differ from the published archive prefix")
    for k in args.precision:
        for b in args.block_size:
            size = compare(events, k, b)
            print(json.dumps(dict(events=len(events), precision=k, block_size=b,
                arithmetic_bytes=len(reference), ideal_model_bytes=ideal_bytes,
                rans_framed_bytes=size, delta=size-len(reference),
                bit_roundtrip=True)))


if __name__ == "__main__":
    main()

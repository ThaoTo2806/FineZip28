"""Prepare the datasets used by the FineZip experiments.

Examples:
    python update/prepare_data.py --enwik8 path/to/enwik8 --output data
    python update/prepare_data.py --enwik8 path/to/enwik8 --output data --mixed

The script writes a byte-exact one-megabyte enwik8 slice and, optionally, a
controlled heterogeneous mixture of wiki text, Python source, and CSV-like rows.
"""
from __future__ import annotations

import argparse
import pathlib
import sysconfig

import numpy as np


def write_enwik8_slice(source: pathlib.Path, output: pathlib.Path, size: int) -> pathlib.Path:
    raw = source.read_bytes()
    if len(raw) < size:
        raise ValueError(f"{source} has {len(raw)} bytes; need at least {size}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw[:size])
    return output


def make_csv_bytes(size: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    rows = ["id,temp,pressure,flag"]
    index = 0
    while len("\n".join(rows).encode("utf-8")) < size:
        rows.append(
            f"{index},{rng.normal(20, 3):.3f},"
            f"{rng.normal(1013, 8):.2f},{int(rng.random() < 0.1)}"
        )
        index += 1
    return ("\n".join(rows) + "\n").encode("utf-8")[:size]


def make_mixed_dataset(source: pathlib.Path, output: pathlib.Path, size: int) -> pathlib.Path:
    raw = source.read_bytes()
    if len(raw) < 400_000:
        raise ValueError("enwik8 source must contain at least 400,000 bytes")

    stdlib = pathlib.Path(sysconfig.get_paths()["stdlib"])
    python_files = sorted(stdlib.rglob("*.py"))
    code = b"".join(path.read_bytes() for path in python_files)[:300_000]
    table = make_csv_bytes(300_000)
    wiki_a = raw[50_000_000:50_200_000]
    wiki_b = raw[50_200_000:50_400_000]

    mixed = b"".join([
        wiki_a,
        code[:150_000],
        table[:150_000],
        wiki_b,
        code[150_000:],
        table[150_000:],
    ])
    text = mixed[:size].decode("utf-8", errors="ignore").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(text)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()

    slice_path = write_enwik8_slice(
        args.enwik8,
        args.output / "enwik8_1mb.txt",
        args.size,
    )
    print(f"enwik8 slice: {slice_path} ({slice_path.stat().st_size} bytes)")

    if args.mixed:
        mixed_path = make_mixed_dataset(
            args.enwik8,
            args.output / "mixed_1mb.txt",
            args.size,
        )
        print(f"mixed dataset: {mixed_path} ({mixed_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

"""
export checkpoint weights to binary format.

usage:
    uv run export.py -e NAME -o DIR
    uv run export.py -e NAME -o DIR --quant-bits 6
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch

from experiments import EXPERIMENTS
from train import load_checkpoint, resolve_groups


def quantize_tensor(tensor: torch.Tensor, scale_max: int = 127) -> tuple[bytes, float]:
    """quantize a float32 tensor with absmax scaling."""
    flat = tensor.detach().cpu().flatten().float()
    absmax = flat.abs().max().item()
    scale = scale_max / absmax if absmax > 0 else 1.0
    quantized = (flat * scale).round().clamp(-scale_max, scale_max).to(torch.int8)
    return bytes(quantized.numpy().tobytes()), scale


def pack_int6(data: bytes) -> bytes:
    """pack signed int8 values (range [-31, 31]) into 6-bit packed bytes.

    4 values (24 bits) → 3 bytes. remainder values packed into final partial bytes.
    bytes are unsigned (0-255), so values >= 128 represent negative signed ints.
    """
    # convert unsigned bytes to 6-bit unsigned: signed_val + 31 → [0, 62]
    u = [(b - 256 + 31) if b >= 128 else (b + 31) for b in data]
    out = bytearray()
    n = len(u)
    i = 0
    while i + 3 < n:
        out.append((u[i] << 2) | (u[i + 1] >> 4))
        out.append(((u[i + 1] & 0x0F) << 4) | (u[i + 2] >> 2))
        out.append(((u[i + 2] & 0x03) << 6) | u[i + 3])
        i += 4
    rem = n - i
    if rem >= 1:
        u1 = u[i + 1] if rem >= 2 else 0
        out.append((u[i] << 2) | (u1 >> 4))
    if rem >= 2:
        u2 = u[i + 2] if rem >= 3 else 0
        out.append(((u1 & 0x0F) << 4) | (u2 >> 2))
    if rem >= 3:
        out.append(((u2 & 0x03) << 6))
    return bytes(out)


def export_weights(
    model: torch.nn.Module,
    ngram_vocabs: dict[str, list[str]],
    langs: list[str],
    output_dir: Path,
    group_name: str,
    scale_max: int = 127,
) -> None:
    """export linear model weights to quantized binary + metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    w_bytes, w_scale = quantize_tensor(model.weight, scale_max)
    b_bytes, b_scale = quantize_tensor(model.bias, scale_max)

    if scale_max == 31:
        w_bytes = pack_int6(w_bytes)
        b_bytes = pack_int6(b_bytes)

    bin_path = output_dir / f"{group_name}.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<2f", w_scale, b_scale))
        f.write(w_bytes)
        f.write(b_bytes)

    meta_path = output_dir / f"{group_name}.json"
    meta = {
        "langs": langs,
        "ngrams": ngram_vocabs,
        "inputSize": model.in_features,
        "outputSize": model.out_features,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

    bin_size = os.path.getsize(bin_path)
    meta_size = os.path.getsize(meta_path)
    print(f"  exported {group_name}: {bin_size / 1024:.1f}KB weights + {meta_size / 1024:.1f}KB meta")


def main() -> None:
    parser = argparse.ArgumentParser(description="export weights from checkpoint")
    parser.add_argument("--experiment", "-e", required=True,
                        help="experiment name to export")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="output directory for weight files")
    parser.add_argument("--quant-bits", type=int, default=8, choices=[6, 8],
                        help="quantization bit width (default: 8)")
    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment: {args.experiment}")
        print(f"available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    groups = resolve_groups(args.experiment)
    results = load_checkpoint(args.experiment, groups)

    scale_max = 31 if args.quant_bits == 6 else 127
    print(f"exporting weights to {args.output}/ (int{args.quant_bits})")
    for group_name, (model, ngram_vocabs, _) in results.items():
        export_weights(model, ngram_vocabs, groups[group_name].langs, args.output, group_name, scale_max)


if __name__ == "__main__":
    main()

"""
export checkpoint weights to binary format.

usage:
    uv run export.py -e NAME -o DIR
    uv run export.py -e NAME -o DIR --quant-bits 6
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import torch

from datasets import NGRAM_TYPES
from experiments import EXPERIMENTS
from train import load_checkpoint, resolve_groups


def quantize_tensor(tensor: torch.Tensor, scale_max: int = 127) -> tuple[bytes, float]:
    """quantize a float32 tensor with absmax scaling."""
    flat = tensor.detach().cpu().flatten().float()
    absmax = flat.abs().max().item()
    scale = scale_max / absmax if absmax > 0 else 1.0
    quantized = (flat * scale).round().clamp(-scale_max, scale_max).to(torch.int8)
    return bytes(quantized.numpy().tobytes()), scale


def quantize_per_row(tensor: torch.Tensor, scale_max: int = 127) -> tuple[bytes, list[float]]:
    """quantize a 2D tensor with per-row absmax scaling for better precision."""
    rows = tensor.detach().cpu().float()
    scales: list[float] = []
    quantized_rows: list[torch.Tensor] = []
    for row in rows:
        absmax = row.abs().max().item()
        scale = scale_max / absmax if absmax > 0 else 1.0
        scales.append(scale)
        q = (row * scale).round().clamp(-scale_max, scale_max).to(torch.int8)
        quantized_rows.append(q)
    quantized = torch.cat(quantized_rows)
    return bytes(quantized.numpy().tobytes()), scales


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


def _encode_strings(strings: list[str]) -> bytes:
    """encode a list of strings as consecutive null-terminated UTF-8."""
    parts = [s.encode("utf-8") + b"\x00" for s in strings]
    return b"".join(parts)


def export_weights(
    model: torch.nn.Module,
    ngram_vocabs: dict[str, list[str]],
    langs: list[str],
    output_dir: Path,
    group_name: str,
    quant_bits: int = 8,
    global_quant: bool = False,
) -> None:
    """export linear model weights + metadata to a single binary file.

    format (v2):
      header: "LD" u8 version(2) u8 quantBits u16le outputSize u16le inputSize
              u16le[4] ngramCounts
      strings: null-terminated UTF-8 (langs then ngrams in type order)
      scales: f32le[outputSize] wScales, f32le bScale
      data: quantized weights (int8 or packed int6), then bias
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_max = 31 if quant_bits == 6 else 127
    output_size = model.out_features
    input_size = model.in_features
    ngram_counts = [len(ngram_vocabs.get(k, [])) for k in NGRAM_TYPES]

    # quantize
    if global_quant:
        w_bytes, w_scale = quantize_tensor(model.weight, scale_max)
        w_scales = [w_scale] * output_size
    else:
        w_bytes, w_scales = quantize_per_row(model.weight, scale_max)
    b_bytes, b_scale = quantize_tensor(model.bias, scale_max)
    if quant_bits == 6:
        w_bytes = pack_int6(w_bytes)
        b_bytes = pack_int6(b_bytes)

    # encode strings
    all_ngrams: list[str] = []
    for k in NGRAM_TYPES:
        all_ngrams.extend(ngram_vocabs.get(k, []))
    langs_bytes = _encode_strings(langs)
    ngrams_bytes = _encode_strings(all_ngrams)

    bin_path = output_dir / f"{group_name}.bin"
    with open(bin_path, "wb") as f:
        # header (16 bytes)
        f.write(b"LD")
        f.write(struct.pack("<2B", 2, quant_bits))
        f.write(struct.pack("<2H", output_size, input_size))
        f.write(struct.pack("<4H", *ngram_counts))

        # strings
        f.write(langs_bytes)
        f.write(ngrams_bytes)

        # scales
        f.write(struct.pack(f"<{output_size}f", *w_scales))
        f.write(struct.pack("<f", b_scale))

        # quantized data
        f.write(w_bytes)
        f.write(b_bytes)

    bin_size = os.path.getsize(bin_path)
    print(f"  exported {group_name}: {bin_size / 1024:.1f}KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="export weights from checkpoint")
    parser.add_argument("--experiment", "-e", required=True,
                        help="experiment name to export")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="output directory for weight files")
    parser.add_argument("--quant-bits", type=int, default=8, choices=[6, 8],
                        help="quantization bit width (default: 8)")
    parser.add_argument("--global-quant", action="store_true",
                        help="use global (per-tensor) instead of per-row weight quantization")
    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment: {args.experiment}")
        print(f"available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    groups = resolve_groups(args.experiment)
    results = load_checkpoint(args.experiment, groups)

    print(f"exporting weights to {args.output}/ (int{args.quant_bits})")
    for group_name, (model, ngram_vocabs, _) in results.items():
        export_weights(model, ngram_vocabs, groups[group_name].langs, args.output, group_name, args.quant_bits, args.global_quant)


if __name__ == "__main__":
    main()

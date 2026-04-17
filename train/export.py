"""
export checkpoint weights to binary format.

product-quantized groups are auto-detected from the checkpoint: if a group's
`.pt` has a `pq_codebook` field (attached by QAT), it's exported as PQ with
null-terminated ngram strings (PQ pins column order to the codebook's subvector
layout). other groups use int6 per-row quantization and prefix-shared ngram
strings.

usage:
    uv run export.py -e NAME -o DIR
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from datasets import NGRAM_TYPES
from experiments import EXPERIMENTS
from pq import PQ_D, PQ_K, pq_assign_indices, pq_encode_weights
from strings_enc import encode_prefix_buckets
from train import load_checkpoint, resolve_groups


# quantBits values for the 1-byte format discriminator in the header
PQ_QUANT = 0xF4
INT6_QUANT = 6

# int6 uses signed values in [-31, 31] (6 bits with one wasted value)
INT6_SCALE_MAX = 31


def quantize_tensor(tensor: torch.Tensor) -> tuple[bytes, float]:
    """quantize a float32 tensor to int6 with absmax scaling."""
    flat = tensor.detach().cpu().flatten().float()
    absmax = flat.abs().max().item()
    scale = INT6_SCALE_MAX / absmax if absmax > 0 else 1.0
    quantized = (flat * scale).round().clamp(-INT6_SCALE_MAX, INT6_SCALE_MAX).to(torch.int8)
    return bytes(quantized.numpy().tobytes()), scale


def quantize_per_row(tensor: torch.Tensor) -> tuple[bytes, list[float]]:
    """quantize a 2D tensor to int6 with per-row absmax scaling for better precision."""
    rows = tensor.detach().cpu().float()
    scales: list[float] = []
    quantized_rows: list[torch.Tensor] = []
    for row in rows:
        absmax = row.abs().max().item()
        scale = INT6_SCALE_MAX / absmax if absmax > 0 else 1.0
        scales.append(scale)
        q = (row * scale).round().clamp(-INT6_SCALE_MAX, INT6_SCALE_MAX).to(torch.int8)
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


def _encode_null_term_strings(strings: list[str]) -> bytes:
    """encode a list of strings as consecutive null-terminated UTF-8."""
    parts = [s.encode("utf-8") + b"\x00" for s in strings]
    return b"".join(parts)


def export_weights(
    model: torch.nn.Module,
    ngram_vocabs: dict[str, list[str]],
    langs: list[str],
    output_dir: Path,
    group_name: str,
    pq_codebook: np.ndarray | None = None,
) -> None:
    """export a group's linear model to a single binary file.

    header (17 bytes): "LD" u8 quantBits u16le outputSize u16le inputSize u16le[5] ngramCounts
    langs: null-terminated UTF-8
    ngrams:
      - PQ (quantBits=0xF4):  null-terminated UTF-8 in logical order
      - int6 (quantBits=6):   nibble-packed prefix-shared per bucket (weight columns
                              permuted to match the lexicographically sorted order)
    scales: f32le[outputSize] wScales + f32le bScale
    weight data:
      - PQ:   u16le K, u8 D, u16le subvectorsPerRow, f32le[K*D] codebook,
              u8[outputSize*subvectorsPerRow] indices
      - int6: packed int6 weights
    bias: packed int6 (always)

    @param pq_codebook fixed codebook from a QAT checkpoint — its presence
      switches the group to PQ encoding. if absent, the group is int6.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_size = model.out_features
    input_size = model.in_features
    ngram_counts = [len(ngram_vocabs.get(k, [])) for k in NGRAM_TYPES]
    use_pq = pq_codebook is not None

    # PQ pins ngram column order to the codebook; int6 is free to reorder
    # columns so we sort each bucket and prefix-share the encoded strings.
    langs_bytes = _encode_null_term_strings(langs)
    if use_pq:
        all_ngrams: list[str] = []
        for k in NGRAM_TYPES:
            all_ngrams.extend(ngram_vocabs.get(k, []))
        ngrams_bytes = _encode_null_term_strings(all_ngrams)
        col_perm = list(range(input_size))
    else:
        buckets = [list(ngram_vocabs.get(k, [])) for k in NGRAM_TYPES]
        ngrams_bytes, perms = encode_prefix_buckets(buckets)
        col_perm = []
        offset = 0
        for bucket, p in zip(buckets, perms, strict=True):
            col_perm.extend(offset + i for i in p)
            offset += len(bucket)

    if col_perm == list(range(input_size)):
        weight = model.weight
    else:
        with torch.no_grad():
            weight = model.weight.detach().cpu()[:, col_perm].contiguous()

    b_bytes, b_scale = quantize_tensor(model.bias)
    b_bytes = pack_int6(b_bytes)

    quant_bits = PQ_QUANT if use_pq else INT6_QUANT
    bin_path = output_dir / f"{group_name}.bin"
    with open(bin_path, "wb") as f:
        f.write(b"LD")
        f.write(struct.pack("<B", quant_bits))
        f.write(struct.pack("<2H", output_size, input_size))
        f.write(struct.pack("<5H", *ngram_counts))
        f.write(langs_bytes)
        f.write(ngrams_bytes)

        if use_pq:
            codebook, indices, w_scales, padded_in, mse = pq_assign_indices(weight, pq_codebook)
            k, d = codebook.shape
            subvectors_per_row = padded_in // d
            print(
                f"  [pq {group_name}] k={k} d={d} padded_in={padded_in} "
                f"subvectors/row={subvectors_per_row} mse={mse:.6f}"
            )

            # row_scale stored as 1/absmax so the TS decoder does
            # weight = centroid / scale (same convention as int6 per-row)
            f.write(w_scales.astype(np.float32).tobytes())
            f.write(struct.pack("<f", b_scale))

            f.write(struct.pack("<H", k))
            f.write(struct.pack("<B", d))
            f.write(struct.pack("<H", subvectors_per_row))
            f.write(codebook.astype(np.float32).tobytes())
            f.write(indices.astype(np.uint8).tobytes())
        else:
            w_bytes, w_scales = quantize_per_row(weight)
            w_bytes = pack_int6(w_bytes)
            f.write(struct.pack(f"<{output_size}f", *w_scales))
            f.write(struct.pack("<f", b_scale))
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
    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment: {args.experiment}")
        print(f"available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    groups = resolve_groups(args.experiment)
    results = load_checkpoint(args.experiment, groups)

    # QAT attaches `pq_codebook` to a group's .pt; presence switches that
    # group to PQ encoding. we already loaded the files via load_checkpoint
    # but those payloads weren't retained, so re-read just the codebook field.
    ckpt_dir = Path(__file__).parent / "checkpoints" / args.experiment
    pq_codebooks: dict[str, np.ndarray] = {}
    for group_name in groups:
        pt_path = ckpt_dir / f"{group_name}.pt"
        if not pt_path.exists():
            continue
        cb = torch.load(pt_path, weights_only=False).get("pq_codebook")
        if cb is not None:
            pq_codebooks[group_name] = np.asarray(cb, dtype=np.float32)

    print(f"exporting weights to {args.output}/")
    for group_name, (model, ngram_vocabs, _) in results.items():
        export_weights(
            model,
            ngram_vocabs,
            groups[group_name].langs,
            args.output,
            group_name,
            pq_codebooks.get(group_name),
        )


if __name__ == "__main__":
    main()

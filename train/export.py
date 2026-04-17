"""
export checkpoint weights to binary format.

product-quantized groups are auto-detected from the checkpoint: if a group's
`.pt` has a `pq_codebook` field (attached by QAT), it's exported as PQ.
otherwise it uses scalar int6/int8 quantization.

usage:
    uv run export.py -e NAME -o DIR
    uv run export.py -e NAME -o DIR --quant-bits 6
    uv run export.py -e NAME -o DIR --pq-groups latin   # override: run fresh k-means
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
from train import load_checkpoint, resolve_groups


# format v4 sentinel: stored in the quantBits byte to signal product quantization.
# decoders distinguish by matching this value instead of a bit width.
PQ_QUANT_SENTINEL = 0xF4


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
    use_pq: bool = False,
    pq_codebook: np.ndarray | None = None,
) -> None:
    """export linear model weights + metadata to a single binary file.

    format (v4):
      header (18 bytes): "LD" u8 version(4) u8 quantBits u16le outputSize u16le inputSize
              u16le[5] ngramCounts
      strings: null-terminated UTF-8 (langs then ngrams in type order)
      scales: f32le[outputSize] wScales, f32le bScale
      data (per quantBits):
        - 6 or 8: packed/raw int weights then bias (v3 layout, carried forward)
        - 0xF4 (PQ): u16le K, u8 D, u16le subvectorsPerRow, then
                     f32le[K*D] codebook, u8[outputSize*subvectorsPerRow] indices,
                     then bias (packed int6)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_size = model.out_features
    input_size = model.in_features
    ngram_counts = [len(ngram_vocabs.get(k, [])) for k in NGRAM_TYPES]

    # encode strings
    all_ngrams: list[str] = []
    for k in NGRAM_TYPES:
        all_ngrams.extend(ngram_vocabs.get(k, []))
    langs_bytes = _encode_strings(langs)
    ngrams_bytes = _encode_strings(all_ngrams)

    # bias always uses int6 path (cheap and consistent)
    b_bytes, b_scale = quantize_tensor(model.bias, 31)
    b_bytes = pack_int6(b_bytes)

    bin_path = output_dir / f"{group_name}.bin"
    with open(bin_path, "wb") as f:
        header_quant_bits = PQ_QUANT_SENTINEL if use_pq else quant_bits
        f.write(b"LD")
        f.write(struct.pack("<2B", 4, header_quant_bits))
        f.write(struct.pack("<2H", output_size, input_size))
        f.write(struct.pack("<5H", *ngram_counts))

        # strings
        f.write(langs_bytes)
        f.write(ngrams_bytes)

        if use_pq:
            if pq_codebook is not None:
                codebook, indices, w_scales, padded_in, mse = pq_assign_indices(
                    model.weight, pq_codebook, PQ_D,
                )
                source = "fixed (QAT)"
            else:
                codebook, indices, w_scales, padded_in, mse = pq_encode_weights(
                    model.weight, k=PQ_K, d=PQ_D, seed=0,
                )
                source = "k-means"
            subvectors_per_row = padded_in // PQ_D
            print(
                f"  [pq {source}] {group_name}: k={PQ_K} d={PQ_D} padded_in={padded_in} "
                f"subvectors/row={subvectors_per_row} mse={mse:.6f}"
            )

            # scales (per-row weight scales + single bias scale)
            # row_scale stored as 1/absmax so the TS decoder can do weight = centroid / scale
            # (mirrors the int6 per-row dequant convention where scale is a divisor).
            f.write(struct.pack(f"<{output_size}f", *w_scales))
            f.write(struct.pack("<f", b_scale))

            # PQ metadata block
            f.write(struct.pack("<H", PQ_K))
            f.write(struct.pack("<B", PQ_D))
            f.write(struct.pack("<H", subvectors_per_row))

            # codebook (K * D float32)
            f.write(codebook.astype(np.float32).tobytes())

            # indices (uint8) in row-major order
            f.write(indices.astype(np.uint8).tobytes())

            # bias (packed int6)
            f.write(b_bytes)
        else:
            # v3-compatible int quant path
            scale_max = 31 if quant_bits == 6 else 127
            if global_quant:
                w_bytes, w_scale = quantize_tensor(model.weight, scale_max)
                w_scales = [w_scale] * output_size
            else:
                w_bytes, w_scales = quantize_per_row(model.weight, scale_max)
            if quant_bits == 6:
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
    parser.add_argument("--quant-bits", type=int, default=None, choices=[6, 8],
                        help="quantization bit width (overrides experiment config)")
    parser.add_argument("--global-quant", action="store_true",
                        help="use global (per-tensor) instead of per-row weight quantization")
    parser.add_argument("--pq-groups", type=str, default="",
                        help="override: force PQ on these comma-separated groups (fresh k-means "
                             "at export time, for checkpoints without a QAT codebook)")
    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment: {args.experiment}")
        print(f"available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    override_pq_groups = {g.strip() for g in args.pq_groups.split(",") if g.strip()}

    exp = EXPERIMENTS[args.experiment]
    groups = resolve_groups(args.experiment)
    results = load_checkpoint(args.experiment, groups)

    # resolve per-group quant bits: CLI flag > experiment config > default (8)
    exp_quant = exp.get("quant_bits", 8)

    # auto-detect PQ from checkpoint: any group whose .pt has `pq_codebook`
    # (attached during QAT) is exported as PQ using the stored codebook.
    ckpt_dir = Path(__file__).parent / "checkpoints" / args.experiment
    fixed_codebooks: dict[str, np.ndarray] = {}
    for group_name in groups:
        pt_path = ckpt_dir / f"{group_name}.pt"
        if pt_path.exists():
            raw = torch.load(pt_path, weights_only=False)
            cb = raw.get("pq_codebook")
            if cb is not None:
                fixed_codebooks[group_name] = np.asarray(cb, dtype=np.float32)
                print(f"  [pq] using fixed codebook from checkpoint for {group_name}")

    print(f"exporting weights to {args.output}/")
    for group_name, (model, ngram_vocabs, _) in results.items():
        if args.quant_bits is not None:
            quant_bits = args.quant_bits
        elif isinstance(exp_quant, dict):
            quant_bits = exp_quant.get(group_name, 8)
        else:
            quant_bits = exp_quant
        use_pq = group_name in fixed_codebooks or group_name in override_pq_groups
        export_weights(
            model,
            ngram_vocabs,
            groups[group_name].langs,
            args.output,
            group_name,
            quant_bits,
            args.global_quant,
            use_pq,
            fixed_codebooks.get(group_name),
        )


if __name__ == "__main__":
    main()

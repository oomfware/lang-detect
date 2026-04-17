"""
average state_dicts of multiple seed checkpoints to collapse init/shuffle variance.

for linear+softmax classifiers, weight averaging is mathematically equivalent to
logit-ensembling at zero inference cost:
    mean(W_i) @ x + mean(b_i) = mean(W_i @ x + b_i)

usage:
    uv run average_seeds.py -o OUTPUT -i SEED1 SEED2 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def average_group(
    ckpt_paths: list[Path],
    output_path: Path,
) -> None:
    """average state_dict tensors element-wise across checkpoints for one script group."""
    ckpts = [torch.load(p, weights_only=False) for p in ckpt_paths]

    ref_langs = ckpts[0]["langs"]
    ref_vocabs = ckpts[0]["ngram_vocabs"]
    for p, c in zip(ckpt_paths[1:], ckpts[1:], strict=True):
        if c["langs"] != ref_langs:
            print(f"langs mismatch: {ckpt_paths[0]} vs {p}")
            sys.exit(1)
        if c["ngram_vocabs"] != ref_vocabs:
            print(f"ngram_vocabs mismatch: {ckpt_paths[0]} vs {p}")
            sys.exit(1)

    # average parameter tensors element-wise
    ref_state = ckpts[0]["state_dict"]
    avg_state: dict[str, torch.Tensor] = {}
    for key, ref_t in ref_state.items():
        stacked = torch.stack([c["state_dict"][key].float() for c in ckpts], dim=0)
        avg_state[key] = stacked.mean(dim=0)

    avg_accuracy = sum(c["accuracy"] for c in ckpts) / len(ckpts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": avg_state,
        "ngram_vocabs": ref_vocabs,
        "langs": ref_langs,
        "accuracy": avg_accuracy,
    }, output_path)

    print(f"  {output_path.name}: averaged {len(ckpts)} seeds, mean test acc {avg_accuracy:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="average seed checkpoints")
    parser.add_argument("--output", "-o", required=True,
                        help="output checkpoint dir name under checkpoints/")
    parser.add_argument("--inputs", "-i", nargs="+", required=True,
                        help="input checkpoint dir names to average")
    args = parser.parse_args()

    ckpt_root = Path(__file__).parent / "checkpoints"
    input_dirs = [ckpt_root / n for n in args.inputs]
    for d in input_dirs:
        if not d.exists():
            print(f"input checkpoint dir not found: {d}")
            sys.exit(1)

    # discover groups from the first input dir
    groups = sorted(p.stem for p in input_dirs[0].glob("*.pt"))
    if not groups:
        print(f"no .pt files found in {input_dirs[0]}")
        sys.exit(1)

    output_dir = ckpt_root / args.output
    print(f"averaging {len(input_dirs)} seeds → {output_dir}/")
    for group in groups:
        ckpt_paths = [d / f"{group}.pt" for d in input_dirs]
        for p in ckpt_paths:
            if not p.exists():
                print(f"missing checkpoint: {p}")
                sys.exit(1)
        average_group(ckpt_paths, output_dir / f"{group}.pt")

    print(f"saved averaged checkpoint to {output_dir}/")


if __name__ == "__main__":
    main()

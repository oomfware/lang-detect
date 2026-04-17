"""
evaluate a saved checkpoint.

usage:
    uv run eval.py -e NAME
"""

from __future__ import annotations

import argparse
import sys

from datasets import DATASET_TRAIN_LIMIT, load_tatoeba, load_udhr
from experiments import EXPERIMENTS
from train import (
    DATASET_TRAIN_PERC,
    UNIQUE_SCRIPT_LANGS,
    collect_nn_langs,
    evaluate,
    load_checkpoint,
    resolve_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate checkpoint")
    parser.add_argument("--experiment", "-e", required=True,
                        help="experiment name to evaluate")
    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment: {args.experiment}")
        print(f"available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    exp = EXPERIMENTS[args.experiment]
    print(f"=== evaluating: {args.experiment} ===")
    print(f"  {exp['description']}")

    groups = resolve_groups(args.experiment)
    results = load_checkpoint(args.experiment, groups)

    unique_script_langs = list(UNIQUE_SCRIPT_LANGS.keys())
    cjk_langs = ["jpn", "cmn"]
    all_langs = list(dict.fromkeys(unique_script_langs + cjk_langs + collect_nn_langs(groups)))

    # UDHR eval
    udhr_data = load_udhr(set(all_langs))
    if udhr_data:
        udhr_sentences: list[tuple[str, str]] = []
        for lang, items in udhr_data.items():
            for datum in items:
                udhr_sentences.append((lang, datum.sentence))

        udhr_acc, udhr_per_lang = evaluate(udhr_sentences, groups, results)
        print(f"\n--- UDHR eval ({len(udhr_sentences)} sentences, {len(udhr_per_lang)} langs) ---")
        print(f"  overall accuracy: {udhr_acc:.2f}%")

        udhr_accs = sorted(udhr_per_lang.items(), key=lambda x: x[1][0])
        for lang, (acc, total) in udhr_accs:
            if acc < 100.0:
                print(f"    {lang}: {acc:.1f}% ({total})")

    # short-string eval
    data_limit = exp.get("train_cfg", {}).get("data_limit", DATASET_TRAIN_LIMIT)
    tatoeba = load_tatoeba(set(all_langs), limit=data_limit)
    for max_chars in [20, 30, 40]:
        short_sentences: list[tuple[str, str]] = []
        for lang in all_langs:
            data = tatoeba.get(lang, [])
            split_idx = int(len(data) * DATASET_TRAIN_PERC)
            for datum in data[split_idx:]:
                s = datum.sentence
                if len(s) > max_chars:
                    cut = s[:max_chars].rfind(" ")
                    s = s[:cut] if cut > 5 else s[:max_chars]
                short_sentences.append((lang, s))

        short_acc, _ = evaluate(short_sentences, groups, results)
        print(f"\n--- short-string eval (≤{max_chars} chars, {len(short_sentences)} sentences) ---")
        print(f"  overall accuracy: {short_acc:.2f}%")


if __name__ == "__main__":
    main()

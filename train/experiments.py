"""experiment configurations for lang-detect training."""

from typing import Any

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "linear_mega_5g": {
        "description": "wide linear with 5-grams for importance source",
        "overrides": {
            "cyrillic": {"unigrams": 500, "bigrams": 700, "trigrams": 700, "quadgrams": 500, "pentagrams": 400, "epochs": 32},
            "arabic": {"unigrams": 200, "bigrams": 300, "trigrams": 300, "quadgrams": 200, "pentagrams": 150, "epochs": 16},
            "devanagari": {"unigrams": 150, "bigrams": 250, "trigrams": 250, "quadgrams": 150, "pentagrams": 100, "epochs": 16},
            "latin": {"unigrams": 500, "bigrams": 900, "trigrams": 900, "quadgrams": 700, "pentagrams": 500},
        },
        "train_cfg": {"label_smoothing": 0.1},
    },
    "rebalanced_5g": {
        "description": "steal bytes from overprovisioned small groups, add 5-grams to Latin (shipped)",
        "prune_from": "linear_mega_5g",
        "overrides": {
            "cyrillic": {"unigrams": 130, "bigrams": 280, "trigrams": 280, "quadgrams": 130},
            "arabic": {"unigrams": 50, "bigrams": 100, "trigrams": 100, "quadgrams": 50},
            "devanagari": {"unigrams": 30, "bigrams": 60, "trigrams": 60, "quadgrams": 30},
            "latin": {"unigrams": 150, "bigrams": 400, "trigrams": 400, "quadgrams": 250, "pentagrams": 100},
        },
        "train_cfg": {"label_smoothing": 0.1, "truncate_aug": 0.75, "focal_gamma": 2.0},
        "quant_bits": 6,
    },
}

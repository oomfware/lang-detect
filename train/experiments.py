"""experiment configurations for lang-detect training."""

from typing import Any

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "linear_mega": {
        "description": "very wide linear for importance source: Lat 3000, Cyr 2400, Ara 1000, Dev 800",
        "overrides": {
            "cyrillic": {"unigrams": 500, "bigrams": 700, "trigrams": 700, "quadgrams": 500, "epochs": 32},
            "arabic": {"unigrams": 200, "bigrams": 300, "trigrams": 300, "quadgrams": 200, "epochs": 16},
            "devanagari": {"unigrams": 150, "bigrams": 250, "trigrams": 250, "quadgrams": 150, "epochs": 16},
            "latin": {"unigrams": 500, "bigrams": 900, "trigrams": 900, "quadgrams": 700},
        },
        "train_cfg": {"label_smoothing": 0.1},
    },
    "pruned_compact_focal": {
        "description": "compact ngrams + focal loss + aug75 + Leipzig (shipped standard)",
        "prune_from": "linear_mega",
        "overrides": {
            "cyrillic": {"unigrams": 150, "bigrams": 300, "trigrams": 300, "quadgrams": 150},
            "arabic": {"unigrams": 80, "bigrams": 140, "trigrams": 140, "quadgrams": 80},
            "devanagari": {"unigrams": 60, "bigrams": 110, "trigrams": 110, "quadgrams": 60},
            "latin": {"unigrams": 150, "bigrams": 400, "trigrams": 400, "quadgrams": 250},
        },
        "train_cfg": {"label_smoothing": 0.1, "truncate_aug": 0.75, "focal_gamma": 2.0},
        "quant_bits": 6,
    },
}

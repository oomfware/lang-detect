"""experiment configurations for lang-detect training."""

from typing import Any

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "linear_mega_5g": {
        "description": "wide linear with 5-grams — phase 1 source for importance selection",
        "overrides": {
            "cyrillic": {"unigrams": 500, "bigrams": 700, "trigrams": 700, "quadgrams": 500, "pentagrams": 400, "epochs": 32},
            "arabic": {"unigrams": 200, "bigrams": 300, "trigrams": 300, "quadgrams": 200, "pentagrams": 150, "epochs": 16},
            "devanagari": {"unigrams": 150, "bigrams": 250, "trigrams": 250, "quadgrams": 150, "pentagrams": 100, "epochs": 16},
            "latin": {"unigrams": 500, "bigrams": 900, "trigrams": 900, "quadgrams": 700, "pentagrams": 500},
        },
        "train_cfg": {"label_smoothing": 0.1},
    },
    "lean_v5": {
        "description": "shipped config — lean ratio + QAT product quantization on latin",
        "prune_from": "linear_mega_5g",
        "seed": 0,
        "overrides": {
            "cyrillic": {"unigrams": 130, "bigrams": 280, "trigrams": 280, "quadgrams": 130},
            "arabic": {"unigrams": 25, "bigrams": 50, "trigrams": 50, "quadgrams": 25},
            "devanagari": {"unigrams": 15, "bigrams": 30, "trigrams": 30, "quadgrams": 15},
            "latin": {"unigrams": 165, "bigrams": 425, "trigrams": 425, "quadgrams": 235, "pentagrams": 90},
        },
        "train_cfg": {"label_smoothing": 0.1, "truncate_aug": 0.75, "focal_gamma": 2.0},
        "qat_pq": {
            "groups": ["latin"],
            "epochs": 20,
            "lr": 0.001,
            "k": 256,
            "d": 4,
            "seed": 0,
        },
    },
}

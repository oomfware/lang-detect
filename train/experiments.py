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
    "pruned_mega": {
        "description": "importance-pruned from mega, best standard candidate (57.4KB, UDHR 95.39%)",
        "prune_from": "linear_mega",
        "train_cfg": {"label_smoothing": 0.1},
    },
    "pruned_mega_aug50": {
        "description": "pruned_mega + 50% truncation augmentation, best short-string candidate",
        "prune_from": "linear_mega",
        "train_cfg": {"label_smoothing": 0.1, "truncate_aug": 0.5},
    },
    "pruned_mega_compact": {
        "description": "importance-pruned from mega to compact sizes, best lite candidate (~43KB)",
        "prune_from": "linear_mega",
        "overrides": {
            "cyrillic": {"unigrams": 150, "bigrams": 300, "trigrams": 300, "quadgrams": 150},
            "arabic": {"unigrams": 80, "bigrams": 140, "trigrams": 140, "quadgrams": 80},
            "devanagari": {"unigrams": 60, "bigrams": 110, "trigrams": 110, "quadgrams": 60},
            "latin": {"unigrams": 150, "bigrams": 400, "trigrams": 400, "quadgrams": 250},
        },
        "train_cfg": {"label_smoothing": 0.1},
    },
    "linear_mega_srp": {
        "description": "mega source with Serbian in both groups + script filtering",
        "overrides": {
            "cyrillic": {"unigrams": 500, "bigrams": 700, "trigrams": 700, "quadgrams": 500, "filter_by_script": True, "epochs": 32},
            "arabic": {"unigrams": 200, "bigrams": 300, "trigrams": 300, "quadgrams": 200, "epochs": 16},
            "devanagari": {"unigrams": 150, "bigrams": 250, "trigrams": 250, "quadgrams": 150, "epochs": 16},
            "latin": {
                "langs": [
                    "afr", "aze", "cat", "ces", "dan", "deu", "eng", "est", "eus", "fin",
                    "fra", "hau", "hrv", "hun", "ind", "isl", "ita", "lit", "nld", "nob",
                    "pol", "por", "ron", "run", "slk", "spa", "srp", "swe", "tgl", "tur", "vie",
                ],
                "unigrams": 500, "bigrams": 900, "trigrams": 900, "quadgrams": 700,
                "filter_by_script": True,
            },
        },
        "train_cfg": {"label_smoothing": 0.1},
    },
    "pruned_mega_srp": {
        "description": "importance-pruned from mega_srp: Serbian in both groups + script filtering",
        "prune_from": "linear_mega_srp",
        "overrides": {
            "cyrillic": {"unigrams": 250, "bigrams": 350, "trigrams": 350, "quadgrams": 250, "filter_by_script": True},
            "arabic": {"unigrams": 100, "bigrams": 150, "trigrams": 150, "quadgrams": 100},
            "devanagari": {"unigrams": 80, "bigrams": 120, "trigrams": 120, "quadgrams": 80},
            "latin": {
                "langs": [
                    "afr", "aze", "cat", "ces", "dan", "deu", "eng", "est", "eus", "fin",
                    "fra", "hau", "hrv", "hun", "ind", "isl", "ita", "lit", "nld", "nob",
                    "pol", "por", "ron", "run", "slk", "spa", "srp", "swe", "tgl", "tur", "vie",
                ],
                "unigrams": 200, "bigrams": 550, "trigrams": 550, "quadgrams": 300,
                "filter_by_script": True,
            },
        },
        "train_cfg": {"label_smoothing": 0.1},
    },
}

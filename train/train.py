"""
lang-detect training pipeline.

usage:
    uv run train.py                    # train default (pruned_mega)
    uv run train.py -e NAME            # run a named experiment
    uv run train.py -l                 # list available experiments
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from experiments import EXPERIMENTS

# ── constants ──

TATOEBA_PATH = Path(__file__).parent / "resources" / "tatoeba.csv"
UDHR_PATH = Path(__file__).parent / "resources" / "udhr"
DATASET_TRAIN_LENGTH_MIN = 40
DATASET_TRAIN_LIMIT = 9000
DATASET_TRAIN_PERC = 0.8

NGRAM_TYPES = ["unigrams", "bigrams", "trigrams", "quadgrams"]

# ── script groups ──

UNIQUE_SCRIPT_LANGS = {
    "kor": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"),
    "kat": re.compile(r"[\u10A0-\u10FF\u2D00-\u2D2F]"),
    "hye": re.compile(r"[\u0530-\u058F]"),
    "ben": re.compile(r"[\u0980-\u09FF]"),
    "ell": re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]"),
    "heb": re.compile(r"[\u0590-\u05FF]"),
}

CJK_JPN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
CJK_CMN = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")


@dataclass
class GroupConfig:
    name: str
    langs: list[str]
    test: re.Pattern[str]
    unigrams: int
    bigrams: int
    trigrams: int
    quadgrams: int
    batch_size: int = 32
    epochs: int = 64
    filter_by_script: bool = False  # filter training data by group's script regex (for dual-script langs)


GROUPS: dict[str, GroupConfig] = {
    "cyrillic": GroupConfig(
        name="cyrillic",
        langs=["bel", "bul", "kaz", "mkd", "rus", "srp", "ukr"],
        test=re.compile(r"[\u0400-\u04FF]"),
        unigrams=250, bigrams=350, trigrams=350, quadgrams=250,
    ),
    "arabic": GroupConfig(
        name="arabic",
        langs=["ara", "ckb", "pes"],
        test=re.compile(r"[\u0600-\u06FF\u0750-\u077F]"),
        unigrams=100, bigrams=150, trigrams=150, quadgrams=100, epochs=32,
    ),
    "devanagari": GroupConfig(
        name="devanagari",
        langs=["hin", "mar"],
        test=re.compile(r"[\u0900-\u097F]"),
        unigrams=80, bigrams=120, trigrams=120, quadgrams=80, epochs=32,
    ),
    "latin": GroupConfig(
        name="latin",
        langs=[
            "afr", "aze", "cat", "ces", "dan", "deu", "eng", "est", "eus", "fin",
            "fra", "hau", "hrv", "hun", "ind", "isl", "ita", "lit", "nld", "nob",
            "pol", "por", "ron", "run", "slk", "spa", "swe", "tgl", "tur", "vie",
        ],
        test=re.compile(r"[a-zA-Z\u00C0-\u024F]"),
        unigrams=200, bigrams=550, trigrams=550, quadgrams=300,
    ),
}


# ── experiment helpers ──

def resolve_groups(experiment_name: str) -> dict[str, GroupConfig]:
    """apply experiment overrides to base GROUPS config."""
    exp = EXPERIMENTS[experiment_name]
    groups = {}
    for name, config in GROUPS.items():
        overrides = exp.get("overrides", {}).get(name, {})
        if overrides:
            groups[name] = GroupConfig(**{**vars(config), **overrides})
        else:
            groups[name] = config
    return groups


def load_checkpoint(
    experiment_name: str,
    groups: dict[str, GroupConfig],
) -> dict[str, tuple[nn.Module, dict[str, list[str]], float]]:
    """load models from a saved checkpoint."""
    ckpt_dir = Path(__file__).parent / "checkpoints" / experiment_name
    if not ckpt_dir.exists():
        print(f"no checkpoint found at {ckpt_dir}/")
        sys.exit(1)

    results: dict[str, tuple[nn.Module, dict[str, list[str]], float]] = {}
    for group_name, config in groups.items():
        ckpt = torch.load(ckpt_dir / f"{group_name}.pt", weights_only=False)
        input_size = sum(len(v) for v in ckpt["ngram_vocabs"].values())
        num_classes = len(ckpt["langs"])
        model = nn.Linear(input_size, num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        results[group_name] = (model, ckpt["ngram_vocabs"], ckpt["accuracy"])

    return results


# ── text normalization ──

HYPHEN_RE = re.compile(r"-+")
MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _is_letter_mark_or_space(c: str) -> bool:
    r"""match JS regex [^\p{L}\p{M}\s] — keep letters, marks, and whitespace."""
    cat = unicodedata.category(c)
    return cat.startswith("L") or cat.startswith("M") or c.isspace()


def normalize(text: str) -> str:
    """normalize text for ngram extraction, matching the JS implementation."""
    text = HYPHEN_RE.sub(" ", text)
    text = "".join(c for c in text if _is_letter_mark_or_space(c))
    text = MULTI_SPACE_RE.sub(" ", text)
    text = text.lower().strip()
    return f" {text} "


@dataclass
class NgramData:
    counts: dict[str, int]
    freqs: dict[str, float]


def extract_ngrams(text: str, n: int) -> NgramData:
    """extract ngram counts and frequencies from normalized text."""
    counts: dict[str, int] = {}
    total = 0
    for i in range(len(text) - n + 1):
        gram = text[i:i + n]
        counts[gram] = counts.get(gram, 0) + 1
        total += 1

    if total == 0:
        return NgramData(counts={}, freqs={})
    freqs = {gram: count / total for gram, count in counts.items()}
    return NgramData(counts=counts, freqs=freqs)


# ── dataset loading ──

@dataclass
class RawDatum:
    lang: str
    sentence: str
    ngrams: dict[str, NgramData]  # {type: NgramData}


def _make_datum(sentence: str) -> RawDatum:
    """create a RawDatum from a raw sentence string."""
    norm = normalize(sentence)
    ngrams = {
        "unigrams": extract_ngrams(norm, 1),
        "bigrams": extract_ngrams(norm, 2),
        "trigrams": extract_ngrams(norm, 3),
        "quadgrams": extract_ngrams(norm, 4),
    }
    return RawDatum(lang="", sentence=sentence, ngrams=ngrams)


def load_tatoeba(langs_set: set[str], limit: int) -> dict[str, list[RawDatum]]:
    """load sentences from the Tatoeba dataset using reservoir sampling for uniform selection."""
    # reservoir sampling: uniformly sample `limit` sentences per language in a single pass.
    # short sentences (< DATASET_TRAIN_LENGTH_MIN) are collected separately as fallback.
    primary: dict[str, list[RawDatum]] = {lang: [] for lang in langs_set}
    primary_seen: dict[str, int] = {lang: 0 for lang in langs_set}
    fallback: dict[str, list[RawDatum]] = {lang: [] for lang in langs_set}
    fallback_seen: dict[str, int] = {lang: 0 for lang in langs_set}

    rng = random.Random(42)

    with open(TATOEBA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            _, lang, sentence = parts
            if lang not in langs_set:
                continue

            is_long = len(sentence) >= DATASET_TRAIN_LENGTH_MIN

            if is_long:
                n = primary_seen[lang]
                primary_seen[lang] = n + 1
                if n < limit:
                    primary[lang].append((sentence, n))
                else:
                    j = rng.randint(0, n)
                    if j < limit:
                        primary[lang][j] = (sentence, n)
            else:
                n = fallback_seen[lang]
                fallback_seen[lang] = n + 1
                if n < limit:
                    fallback[lang].append((sentence, n))
                else:
                    j = rng.randint(0, n)
                    if j < limit:
                        fallback[lang][j] = (sentence, n)

    # convert to RawDatum and fill from fallback if needed
    result: dict[str, list[RawDatum]] = {}
    for lang in langs_set:
        data = [_make_datum(s) for s, _ in primary[lang]]
        for d in data:
            d.lang = lang
        needed = max(0, limit - len(data))
        if needed > 0:
            fb = [_make_datum(s) for s, _ in fallback[lang][:needed]]
            for d in fb:
                d.lang = lang
            data.extend(fb)
        result[lang] = data

    return result


# curated mapping from UDHR file codes to our language codes.
# only standard/modern variants — excludes dialects (068=Welche), historical orthographies,
# and non-standard scripts (vie_han, tgl_tglg).
UDHR_CODE_TO_LANG: dict[str, str] = {
    "afr": "afr",
    "bel": "bel",
    "ben": "ben",
    "bul": "bul",
    "cat": "cat",
    "ces": "ces",
    "ckb": "ckb",
    "cmn_hans": "cmn",
    "dan": "dan",
    "deu_1996": "deu",
    "ell_monotonic": "ell",
    "eng": "eng",
    "eus": "eus",
    "fin": "fin",
    "fra": "fra",
    "hau_NG": "hau",
    "heb": "heb",
    "hin": "hin",
    "hrv": "hrv",
    "hun": "hun",
    "hye": "hye",
    "ind": "ind",
    "isl": "isl",
    "ita": "ita",
    "jpn": "jpn",
    "kat": "kat",
    "kaz": "kaz",
    "kor": "kor",
    "lit": "lit",
    "mar": "mar",
    "mkd": "mkd",
    "nld": "nld",
    "nob": "nob",
    "pes_1": "pes",
    "pol": "pol",
    "por_BR": "por",
    "por_PT": "por",
    "ron_2006": "ron",
    "run": "run",
    "rus": "rus",
    "slk": "slk",
    "spa": "spa",
    "srp_cyrl": "srp",
    "srp_latn": "srp",
    "swe": "swe",
    "tgl": "tgl",
    "tur": "tur",
    "ukr": "ukr",
    "vie": "vie",
}


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_udhr(langs_set: set[str]) -> dict[str, list[RawDatum]]:
    """load paragraphs from the UDHR HTML declarations."""
    decl_dir = UDHR_PATH / "declaration"
    if not decl_dir.exists():
        return {}

    result: dict[str, list[RawDatum]] = {}

    for code, lang in UDHR_CODE_TO_LANG.items():
        if lang not in langs_set:
            continue

        html_file = decl_dir / f"{code}.html"
        if not html_file.exists():
            continue

        content = html_file.read_text(encoding="utf-8")
        for match in re.finditer(r"<p>(.*?)</p>", content, re.DOTALL):
            text = _HTML_TAG_RE.sub("", match.group(1)).strip()
            if len(text) < 10:
                continue
            datum = _make_datum(text)
            datum.lang = lang
            result.setdefault(lang, []).append(datum)

    return result


def load_dataset_raw(langs: list[str], limit: int = DATASET_TRAIN_LIMIT) -> dict[str, list[RawDatum]]:
    """load and preprocess training data from Tatoeba + UDHR."""
    langs_set = set(langs)

    # load from both sources
    tatoeba = load_tatoeba(langs_set, limit)
    udhr = load_udhr(langs_set)

    # merge: Tatoeba is primary, UDHR supplements
    result: dict[str, list[RawDatum]] = {}
    for lang in langs:
        t_data = tatoeba.get(lang, [])
        u_data = udhr.get(lang, [])
        result[lang] = t_data + u_data

    return result


# ── ngram selection ──

def select_ngrams_roundrobin(
    dataset: dict[str, list[RawDatum]],
    langs: list[str],
    limit: int,
    ngram_type: str,
) -> list[str]:
    """select top ngrams using round-robin across languages for coverage."""
    # aggregate raw counts per language (matching original's sum of ngram.count)
    lang_counts: dict[str, Counter[str]] = {}
    for lang in langs:
        lang_counts[lang] = Counter()
        for datum in dataset.get(lang, []):
            ngram_data = datum.ngrams.get(ngram_type)
            if ngram_data is None:
                continue
            for gram, count in ngram_data.counts.items():
                lang_counts[lang][gram] += count

    # sort each language's ngrams by frequency (descending)
    per_lang_sorted: list[list[str]] = []
    for lang in langs:
        sorted_grams = [g for g, _ in lang_counts[lang].most_common()]
        per_lang_sorted.append(sorted_grams)

    # round-robin selection
    selected: list[str] = []
    selected_set: set[str] = set()

    while len(selected) < limit:
        added_any = False
        for lang_grams in per_lang_sorted:
            if len(selected) >= limit:
                break
            while lang_grams:
                gram = lang_grams.pop(0)
                if gram not in selected_set:
                    selected.append(gram)
                    selected_set.add(gram)
                    added_any = True
                    break
            # if this language is exhausted, skip it
        if not added_any:
            break

    # pad with empty strings if needed (matches original behavior)
    while len(selected) < limit:
        selected.append("")

    return selected[:limit]


def select_ngrams_by_importance(
    model: nn.Module,
    ngram_vocabs: dict[str, list[str]],
    target_sizes: dict[str, int],
) -> dict[str, list[str]]:
    """select top ngrams per type by weight importance from a trained model.

    uses L1 norm of each ngram's weight vector (sum of absolute weights
    across all output classes) as the importance metric. this selects ngrams
    that the model found most discriminative, rather than most frequent.
    """
    w = model.weight.detach()
    importance = w.abs().sum(dim=0)

    new_vocabs = {}
    offset = 0
    for ngram_type in NGRAM_TYPES:
        vocab = ngram_vocabs.get(ngram_type)
        if not vocab:
            continue
        n = len(vocab)
        scores = importance[offset:offset + n]
        target = min(target_sizes.get(ngram_type, n), n)

        _, top_indices = scores.topk(target)
        top_indices_sorted = top_indices.sort().values
        new_vocabs[ngram_type] = [vocab[i.item()] for i in top_indices_sorted]
        offset += n

    return new_vocabs


# ── dataset preparation ──

def _truncate_datum(datum: RawDatum, max_chars: int) -> RawDatum:
    """create a truncated copy of a datum for short-string augmentation."""
    s = datum.sentence
    if len(s) <= max_chars:
        return datum
    # snap to word boundary
    cut = s[:max_chars].rfind(" ")
    s = s[:cut] if cut > 5 else s[:max_chars]
    result = _make_datum(s)
    result.lang = datum.lang
    return result


def prepare_dataset(
    raw: dict[str, list[RawDatum]],
    langs: list[str],
    config: GroupConfig,
    preset_vocabs: dict[str, list[str]] | None = None,
    truncate_aug: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[str]]]:
    """
    prepare train/test tensors for a group.

    truncate_aug: fraction of training data to duplicate as truncated copies (0.0 = none).
    returns: (train_x, train_y, test_x, test_y, ngram_vocabs)
    """
    if preset_vocabs is not None:
        ngram_vocabs = preset_vocabs
    else:
        ngram_vocabs = {
            "unigrams": select_ngrams_roundrobin(raw, langs, config.unigrams, "unigrams"),
            "bigrams": select_ngrams_roundrobin(raw, langs, config.bigrams, "bigrams"),
            "trigrams": select_ngrams_roundrobin(raw, langs, config.trigrams, "trigrams"),
            "quadgrams": select_ngrams_roundrobin(raw, langs, config.quadgrams, "quadgrams"),
        }

    train_inputs: list[list[float]] = []
    train_labels: list[int] = []
    test_inputs: list[list[float]] = []
    test_labels: list[int] = []

    rng = random.Random(42)

    for lang in langs:
        data = raw.get(lang, [])
        if config.filter_by_script:
            data = [d for d in data if config.test.search(d.sentence)]
        lang_idx = langs.index(lang)

        split_idx = int(len(data) * DATASET_TRAIN_PERC)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        for datum in train_data:
            train_inputs.append(_build_feature_vector(datum, ngram_vocabs))
            train_labels.append(lang_idx)

        if truncate_aug > 0:
            n_aug = int(len(train_data) * truncate_aug)
            aug_indices = rng.sample(range(len(train_data)), min(n_aug, len(train_data)))
            for i in aug_indices:
                max_chars = rng.randint(15, 40)
                trunc = _truncate_datum(train_data[i], max_chars)
                train_inputs.append(_build_feature_vector(trunc, ngram_vocabs))
                train_labels.append(lang_idx)

        for datum in test_data:
            test_inputs.append(_build_feature_vector(datum, ngram_vocabs))
            test_labels.append(lang_idx)

    train_x = torch.tensor(train_inputs, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = torch.tensor(test_inputs, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    return train_x, train_y, test_x, test_y, ngram_vocabs


def _build_feature_vector(datum: RawDatum, ngram_vocabs: dict[str, list[str]]) -> list[float]:
    """build the input feature vector for a datum given ngram vocabularies."""
    vec: list[float] = []
    for ngram_type in NGRAM_TYPES:
        vocab = ngram_vocabs.get(ngram_type)
        if not vocab:
            continue
        ngram_data = datum.ngrams.get(ngram_type)
        freqs = ngram_data.freqs if ngram_data else {}
        vec.extend(freqs.get(gram, 0.0) for gram in vocab)
    return vec


# ── training ──

@dataclass
class TrainConfig:
    """training hyperparameters beyond the group architecture."""
    lr: float = 0.001
    label_smoothing: float = 0.0
    data_limit: int = DATASET_TRAIN_LIMIT  # sentences per language
    truncate_aug: float = 0.0  # fraction of training data to duplicate as truncated (15-40 char) copies


def train_group(
    config: GroupConfig,
    raw: dict[str, list[RawDatum]],
    train_cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
    preset_vocabs: dict[str, list[str]] | None = None,
) -> tuple[nn.Module, dict[str, list[str]], float]:
    """
    train a linear model for a single script group.

    returns: (model, ngram_vocabs, test_accuracy)
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    langs = config.langs
    if verbose:
        print(f"\n--- {config.name} ({len(langs)} langs, {config.epochs} epochs, batch {config.batch_size}) ---")

    train_x, train_y, test_x, test_y, ngram_vocabs = prepare_dataset(
        raw, langs, config, preset_vocabs, truncate_aug=train_cfg.truncate_aug,
    )
    if verbose:
        print(f"  train: {len(train_x)}, test: {len(test_x)}")

    input_size = sum(len(v) for v in ngram_vocabs.values())
    model = nn.Linear(input_size, len(langs))

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        if verbose and (epoch + 1) % max(1, config.epochs // 4) == 0:
            print(f"  epoch {epoch + 1}/{config.epochs} — loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        preds = logits.argmax(dim=1)
        correct = (preds == test_y).sum().item()
        accuracy = correct / len(test_y) * 100

    if verbose:
        print(f"  accuracy: {accuracy:.2f}%")

    return model, ngram_vocabs, accuracy


# ── evaluation ──

def detect_language(
    sentence: str,
    groups: dict[str, GroupConfig],
    results: dict[str, tuple[nn.Module, dict[str, list[str]], float]],
) -> str | None:
    """run tiered language detection on a single sentence."""
    # unique script detection
    for script_lang, pattern in UNIQUE_SCRIPT_LANGS.items():
        if pattern.search(sentence):
            return script_lang

    # CJK detection
    if CJK_JPN.search(sentence):
        return "jpn"
    if CJK_CMN.search(sentence):
        return "cmn"

    # NN group detection
    for group_name, config in groups.items():
        if config.test.search(sentence):
            model, ngram_vocabs, _ = results[group_name]
            return _infer(model, config.langs, ngram_vocabs, sentence)

    # fallback to latin
    model, ngram_vocabs, _ = results["latin"]
    return _infer(model, groups["latin"].langs, ngram_vocabs, sentence)


def evaluate(
    test_sentences: list[tuple[str, str]],
    groups: dict[str, GroupConfig],
    results: dict[str, tuple[nn.Module, dict[str, list[str]], float]],
) -> tuple[float, dict[str, tuple[float, int]]]:
    """
    evaluate detection accuracy on a list of (expected_lang, sentence) pairs.

    returns: (overall_accuracy_pct, {lang: (accuracy_pct, count)})
    """
    per_lang_correct: dict[str, int] = {}
    per_lang_total: dict[str, int] = {}

    for lang, sentence in test_sentences:
        per_lang_total[lang] = per_lang_total.get(lang, 0) + 1
        detected = detect_language(sentence, groups, results)
        if detected == lang:
            per_lang_correct[lang] = per_lang_correct.get(lang, 0) + 1

    total = len(test_sentences)
    passed = sum(per_lang_correct.values())
    overall_acc = passed / total * 100 if total > 0 else 0.0

    per_lang: dict[str, tuple[float, int]] = {}
    for lang, count in per_lang_total.items():
        correct = per_lang_correct.get(lang, 0)
        per_lang[lang] = (correct / count * 100, count)

    return overall_acc, per_lang


def _infer(
    model: nn.Module,
    langs: list[str],
    ngram_vocabs: dict[str, list[str]],
    text: str,
) -> str:
    """run inference on a single text, return the predicted language."""
    datum = _make_datum(text)
    vec = _build_feature_vector(datum, ngram_vocabs)
    x = torch.tensor([vec], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return langs[pred]


# ── full pipeline ──

def _calc_binary_size(model: nn.Module) -> int:
    """calculate the quantized binary weight size for a model."""
    num_params = len(list(model.parameters()))
    total_weights = sum(p.numel() for p in model.parameters())
    return total_weights + num_params * 4  # scale floats × 4 bytes each


def run_full_pipeline(
    groups: dict[str, GroupConfig],
    train_cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
    preset_vocabs: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, tuple[nn.Module, dict[str, list[str]], float]]:
    """
    train all groups and run end-to-end evaluation.

    returns: {group_name: (model, ngram_vocabs, accuracy)}
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    # collect all languages for loading (deduplicate for langs in multiple groups)
    all_nn_langs = list(dict.fromkeys(
        lang for g in groups.values() for lang in g.langs
    ))

    unique_script_langs = list(UNIQUE_SCRIPT_LANGS.keys())
    cjk_langs = ["jpn", "cmn"]
    all_langs = list(dict.fromkeys(unique_script_langs + cjk_langs + all_nn_langs))

    if verbose:
        print(f"loading dataset for {len(all_nn_langs)} NN languages "
              f"+ {len(unique_script_langs) + len(cjk_langs)} script-detected...")
        t0 = time.time()

    raw = load_dataset_raw(all_nn_langs, limit=train_cfg.data_limit)

    if verbose:
        dt = time.time() - t0
        print(f"dataset loaded in {dt:.1f}s")

    # train each group
    results: dict[str, tuple[nn.Module, dict[str, list[str]], float]] = {}
    total_params = 0

    for group_name, config in groups.items():
        group_vocabs = preset_vocabs.get(group_name) if preset_vocabs else None
        t0 = time.time()
        model, ngram_vocabs, accuracy = train_group(
            config, raw, train_cfg, verbose=verbose, preset_vocabs=group_vocabs,
        )
        dt = time.time() - t0

        params = sum(p.numel() for p in model.parameters())
        total_params += params
        results[group_name] = (model, ngram_vocabs, accuracy)

        if verbose:
            print(f"  params: {params:,} — trained in {dt:.1f}s")

    # end-to-end evaluation
    if verbose:
        print(f"\n--- full end-to-end test ---")

    full_raw = load_dataset_raw(all_langs, limit=train_cfg.data_limit)
    test_sentences: list[tuple[str, str]] = []
    for lang in all_langs:
        data = full_raw.get(lang, [])
        split_idx = int(len(data) * DATASET_TRAIN_PERC)
        for datum in data[split_idx:]:
            test_sentences.append((lang, datum.sentence))

    overall_acc, per_lang = evaluate(test_sentences, groups, results)

    if verbose:
        total = len(test_sentences)
        passed = int(overall_acc / 100 * total)
        print(f"\n  overall accuracy: {overall_acc:.2f}%")
        print(f"  pass: {passed}, fail: {total - passed}, total: {total}")
        print(f"  total params: {total_params:,}")

        # weight sizes
        total_binary = 0
        for group_name, (model, _, _) in results.items():
            size = _calc_binary_size(model)
            total_binary += size
        print(f"  total int8 binary: {total_binary / 1024:.1f}KB")

        # per-language accuracy (sorted by accuracy ascending to show weakest first)
        print(f"\n  per-language accuracy:")
        lang_accs = sorted(per_lang.items(), key=lambda x: x[1][0])
        for lang, (acc, total) in lang_accs:
            print(f"    {lang}: {acc:.1f}% ({total})")

    # fixed UDHR evaluation (comparable across experiments)
    if verbose:
        udhr_data = load_udhr(set(all_langs))
        if udhr_data:
            udhr_sentences: list[tuple[str, str]] = []
            for lang, items in udhr_data.items():
                for datum in items:
                    udhr_sentences.append((lang, datum.sentence))

            udhr_acc, udhr_per_lang = evaluate(udhr_sentences, groups, results)
            print(f"\n--- fixed UDHR eval ({len(udhr_sentences)} sentences, {len(udhr_per_lang)} langs) ---")
            print(f"  overall accuracy: {udhr_acc:.2f}%")

            udhr_accs = sorted(udhr_per_lang.items(), key=lambda x: x[1][0])
            for lang, (acc, total) in udhr_accs:
                if acc < 100.0:
                    print(f"    {lang}: {acc:.1f}% ({total})")

    return results


# ── CLI ──

def main() -> None:
    parser = argparse.ArgumentParser(description="lang-detect training")
    parser.add_argument("--experiment", "-e", default="pruned_mega",
                        help="experiment name to run")
    parser.add_argument("--list", "-l", action="store_true",
                        help="list available experiments")
    args = parser.parse_args()

    if args.list:
        print("available experiments:")
        for name, exp in EXPERIMENTS.items():
            ckpt_dir = Path(__file__).parent / "checkpoints" / name
            marker = " [checkpoint]" if ckpt_dir.exists() else ""
            print(f"  {name}: {exp['description']}{marker}")
        return

    exp = EXPERIMENTS.get(args.experiment)
    if exp is None:
        print(f"unknown experiment: {args.experiment}")
        print(f"available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    print(f"=== experiment: {args.experiment} ===")
    print(f"  {exp['description']}")

    groups = resolve_groups(args.experiment)
    train_cfg = TrainConfig(**exp.get("train_cfg", {}))

    t0 = time.time()

    if "prune_from" in exp:
        # two-phase: train wide model, prune by importance, retrain
        wide_exp = EXPERIMENTS[exp["prune_from"]]
        wide_groups = {}
        for name, config in GROUPS.items():
            overrides = wide_exp.get("overrides", {}).get(name, {})
            wide_groups[name] = GroupConfig(**{**vars(config), **overrides}) if overrides else config
        wide_train_cfg = TrainConfig(**wide_exp.get("train_cfg", {}))

        print("\n=== phase 1: training wide models for importance analysis ===")
        all_nn_langs = list(dict.fromkeys(
            lang for g in wide_groups.values() for lang in g.langs
        ))
        raw = load_dataset_raw(all_nn_langs, limit=train_cfg.data_limit)

        preset_vocabs: dict[str, dict[str, list[str]]] = {}
        for group_name, config in wide_groups.items():
            model, vocabs, acc = train_group(config, raw, wide_train_cfg)

            target = groups[group_name]
            target_sizes = {
                "unigrams": target.unigrams,
                "bigrams": target.bigrams,
                "trigrams": target.trigrams,
                "quadgrams": target.quadgrams,
            }
            pruned = select_ngrams_by_importance(model, vocabs, target_sizes)

            preset_vocabs[group_name] = pruned
            print(f"  {group_name} pruned:")
            for t in NGRAM_TYPES:
                if t in vocabs and t in pruned:
                    print(f"    {t}: {len(vocabs[t])} → {len(pruned[t])}")

        print("\n=== phase 2: retraining with importance-selected ngrams ===")
        results = run_full_pipeline(groups, train_cfg, preset_vocabs=preset_vocabs)
    else:
        results = run_full_pipeline(groups, train_cfg)

    total_time = time.time() - t0
    print(f"\ntotal time: {total_time:.1f}s")

    # save checkpoint
    ckpt_dir = Path(__file__).parent / "checkpoints" / args.experiment
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for group_name, (model, ngram_vocabs, accuracy) in results.items():
        torch.save({
            "state_dict": model.state_dict(),
            "ngram_vocabs": ngram_vocabs,
            "langs": groups[group_name].langs,
            "accuracy": accuracy,
        }, ckpt_dir / f"{group_name}.pt")
    print(f"checkpoint saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()

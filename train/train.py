"""
lang-detect training pipeline.

usage:
    uv run train.py -e NAME            # run a named experiment
    uv run train.py -l                 # list available experiments
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from datasets import (
    DATASET_TRAIN_LIMIT,
    NGRAM_TYPES,
    RawDatum,
    load_dataset_raw,
    make_datum,
)
from experiments import EXPERIMENTS
from pq import PQ_D, PQ_K, pq_encode_weights, pq_snap_weights

# ── constants ──

DATASET_TRAIN_PERC = 0.8

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

# unique character markers within the Cyrillic script group.
# these characters appear exclusively in one language of our Cyrillic group,
# enabling quick identification before falling through to the NN classifier.
CYRILLIC_UNIQUE_CHARS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\u0402\u0452\u040B\u045B]"), "srp"),   # Ђђ Ћћ
    (re.compile(r"[\u0403\u0453\u040C\u045C\u0405\u0455]"), "mkd"),  # Ѓѓ Ќќ Ѕѕ
    (re.compile(r"[\u040E\u045E]"), "bel"),                # Ўў
    (re.compile(r"[\u0407\u0457\u0404\u0454]"), "ukr"),    # Її Єє
    (re.compile(r"[\u04D8\u04D9\u0492\u0493\u049A\u049B\u04A2\u04A3\u04E8\u04E9\u04AE\u04AF]"), "kaz"),  # Әә Ғғ Ққ Ңң Өө Үү
]


@dataclass
class GroupConfig:
    name: str
    langs: list[str]
    test: re.Pattern[str]
    unigrams: int
    bigrams: int
    trigrams: int
    quadgrams: int
    pentagrams: int = 0
    batch_size: int = 32
    epochs: int = 64


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
    result = make_datum(s)
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
            "pentagrams": select_ngrams_roundrobin(raw, langs, config.pentagrams, "pentagrams"),
        }

    train_inputs: list[list[float]] = []
    train_labels: list[int] = []
    test_inputs: list[list[float]] = []
    test_labels: list[int] = []

    rng = random.Random(42)

    for lang in langs:
        data = raw.get(lang, [])
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


def _build_feature_vector(
    datum: RawDatum,
    ngram_vocabs: dict[str, list[str]],
) -> list[float]:
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

class FocalLoss(nn.Module):
    """focal loss: down-weights easy examples to focus on hard/misclassified ones.

    FL(p) = -alpha * (1 - p)^gamma * log(p)
    when gamma=0, equivalent to cross-entropy. gamma=2 is standard.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = torch.zeros_like(log_probs)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0)

        # focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs).pow(self.gamma)
        loss = -(focal_weight * smooth * log_probs).sum(dim=1)
        return loss.mean()


@dataclass
class TrainConfig:
    """training hyperparameters beyond the group architecture."""
    lr: float = 0.001
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0  # focal loss gamma (0.0 = standard cross-entropy)
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
    """train a linear model for a single script group."""
    if train_cfg is None:
        train_cfg = TrainConfig()

    langs = config.langs
    if verbose:
        print(f"\n--- {config.name} ({len(langs)} langs, {config.epochs} epochs, batch {config.batch_size}) ---")

    train_x, train_y, test_x, test_y, ngram_vocabs = prepare_dataset(
        raw, langs, config, preset_vocabs,
        truncate_aug=train_cfg.truncate_aug,
    )
    if verbose:
        print(f"  train: {len(train_x)}, test: {len(test_x)}")

    input_size = sum(len(v) for v in ngram_vocabs.values())
    model = nn.Linear(input_size, len(langs))

    if train_cfg.focal_gamma > 0:
        criterion = FocalLoss(gamma=train_cfg.focal_gamma, label_smoothing=train_cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=config.batch_size, shuffle=True,
    )

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


# ── QAT for product quantization ──

def qat_group_pq(
    config: GroupConfig,
    raw: dict[str, list[RawDatum]],
    model: nn.Linear,
    ngram_vocabs: dict[str, list[str]],
    train_cfg: TrainConfig,
    epochs: int,
    lr: float,
    k: int,
    d: int,
    seed: int,
) -> tuple[nn.Linear, np.ndarray, float]:
    """fine-tune a group's linear model with PQ applied via straight-through estimator.

    the codebook is fit once on the incoming weights and frozen during training;
    QAT pulls weights toward codebook-friendly positions without needing to re-fit.
    the best snapped test-accuracy epoch is kept (snap-induced noise makes the last
    epoch an unreliable pick).

    @param config group config (langs, batch size, etc.)
    @param raw loaded raw dataset dict
    @param model trained phase-2 linear model (weights are mutated in-place)
    @param ngram_vocabs preset vocabs used during phase 2 — reused here
    @param train_cfg training config (focal gamma, label smoothing, truncate_aug)
    @param epochs number of QAT epochs
    @param lr learning rate
    @param k codebook size
    @param d subvector length
    @param seed k-means + data-shuffle seed
    @returns (tuned model, fixed codebook [k,d] f32, best snapped test accuracy %)
    """
    torch.manual_seed(seed)

    train_x, train_y, test_x, test_y, _ = prepare_dataset(
        raw, config.langs, config, preset_vocabs=ngram_vocabs,
        truncate_aug=train_cfg.truncate_aug,
    )

    # fit codebook once on the phase-2 weights
    centroids_np, _, _, _, init_mse = pq_encode_weights(model.weight, k=k, d=d, seed=seed)
    codebook = torch.from_numpy(centroids_np).float()
    print(f"  [qat-pq {config.name}] initial codebook mse: {init_mse:.6f}")

    # baseline snapped acc (what shipping without QAT would look like)
    model.eval()
    with torch.no_grad():
        orig_w = model.weight.data.clone()
        model.weight.data = pq_snap_weights(model.weight, codebook, d)
        base_snap_acc = (model(test_x).argmax(dim=1) == test_y).float().mean().item() * 100
        model.weight.data = orig_w

    if train_cfg.focal_gamma > 0:
        criterion = FocalLoss(gamma=train_cfg.focal_gamma, label_smoothing=train_cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=config.batch_size, shuffle=True,
    )

    best_snap_acc = base_snap_acc
    best_state = {s_key: s_val.clone() for s_key, s_val in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # straight-through estimator: snap on forward, identity on backward
            w_q = pq_snap_weights(model.weight, codebook, d)
            w_eff = model.weight + (w_q - model.weight).detach()
            logits = F.linear(batch_x, w_eff, model.bias)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        # eval with deployment-identical snapped weights
        model.eval()
        with torch.no_grad():
            orig_w = model.weight.data.clone()
            model.weight.data = pq_snap_weights(model.weight, codebook, d)
            snap_acc = (model(test_x).argmax(dim=1) == test_y).float().mean().item() * 100
            model.weight.data = orig_w

        if snap_acc > best_snap_acc:
            best_snap_acc = snap_acc
            best_state = {s_key: s_val.clone() for s_key, s_val in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  [qat-pq {config.name}] epoch {epoch + 1}/{epochs} — "
                  f"loss {epoch_loss / batches:.4f} — snapped {snap_acc:.2f}%")

    model.load_state_dict(best_state)
    print(f"  [qat-pq {config.name}] best snapped acc: {best_snap_acc:.2f}% "
          f"(baseline snapped: {base_snap_acc:.2f}%)")
    return model, centroids_np, best_snap_acc


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
            # unique character pre-filtering within Cyrillic
            if group_name == "cyrillic":
                for pattern, lang in CYRILLIC_UNIQUE_CHARS:
                    if pattern.search(sentence):
                        return lang
            model, ngram_vocabs, _ = results[group_name]
            return _infer(model, config, ngram_vocabs, sentence)

    # fallback to latin
    model, ngram_vocabs, _ = results["latin"]
    return _infer(model, groups["latin"], ngram_vocabs, sentence)


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
    config: GroupConfig,
    ngram_vocabs: dict[str, list[str]],
    text: str,
) -> str:
    """run inference on a single text, return the predicted language."""
    datum = make_datum(text)
    vec = _build_feature_vector(datum, ngram_vocabs)
    x = torch.tensor([vec], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return config.langs[pred]


# ── full pipeline ──

def _calc_binary_size(
    model: nn.Module,
    ngram_vocabs: dict[str, list[str]],
    langs: list[str],
    quant_bits: int = 6,
) -> int:
    """calculate the exported binary size for a model."""
    output_size = model.out_features
    input_size = model.in_features
    header = 18
    lang_bytes = sum(len(l.encode("utf-8")) + 1 for l in langs)
    ngram_bytes = sum(len(g.encode("utf-8")) + 1 for v in ngram_vocabs.values() for g in v)
    scales = output_size * 4 + 4
    total_values = output_size * input_size + output_size
    if quant_bits == 6:
        data = (total_values * 6 + 7) // 8
    else:
        data = total_values
    return header + lang_bytes + ngram_bytes + scales + data


def run_full_pipeline(
    groups: dict[str, GroupConfig],
    train_cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
    preset_vocabs: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, tuple[nn.Module, dict[str, list[str]], float]]:
    """train all groups."""
    if train_cfg is None:
        train_cfg = TrainConfig()

    # collect all languages for loading (deduplicate for langs in multiple groups)
    all_nn_langs = list(dict.fromkeys(
        lang for g in groups.values() for lang in g.langs
    ))

    if verbose:
        print(f"loading dataset for {len(all_nn_langs)} NN languages...")
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

    if verbose:
        total_binary = sum(
            _calc_binary_size(m, v, groups[g].langs)
            for g, (m, v, _) in results.items()
        )
        print(f"\n  total params: {total_params:,}")
        print(f"  estimated int6 binary: {total_binary / 1024:.1f}KB")

    return results


# ── CLI ──

def _load_preset_vocabs(preset_name: str) -> dict[str, dict[str, list[str]]]:
    """load ngram_vocabs from a named checkpoint to reuse as a fixed preset."""
    ckpt_dir = Path(__file__).parent / "checkpoints" / preset_name
    if not ckpt_dir.exists():
        print(f"no checkpoint found at {ckpt_dir}/ to use as preset")
        sys.exit(1)

    preset: dict[str, dict[str, list[str]]] = {}
    for pt in sorted(ckpt_dir.glob("*.pt")):
        ckpt = torch.load(pt, weights_only=False)
        preset[pt.stem] = ckpt["ngram_vocabs"]
    return preset


def main() -> None:
    parser = argparse.ArgumentParser(description="lang-detect training")
    parser.add_argument("--experiment", "-e", required=True,
                        help="experiment name to run")
    parser.add_argument("--list", "-l", action="store_true",
                        help="list available experiments")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for torch and stdlib RNG (controls nn.Linear init + DataLoader shuffle)")
    parser.add_argument("--preset-from", type=str, default=None,
                        help="load ngram_vocabs from this checkpoint dir and skip phase 1")
    parser.add_argument("--output-name", type=str, default=None,
                        help="override checkpoint output dir name (defaults to experiment name)")
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

    if args.seed is not None:
        # seed before any dataloader/model init so shuffle + nn.Linear init are deterministic.
        # dataset split and truncate_aug use random.Random(42) internally and stay fixed.
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print(f"  seed: {args.seed}")

    if args.preset_from is not None:
        print(f"\n=== loading ngram_vocabs preset from checkpoint: {args.preset_from} ===")
        preset_vocabs = _load_preset_vocabs(args.preset_from)
        for group_name in groups:
            if group_name not in preset_vocabs:
                print(f"preset checkpoint {args.preset_from} missing group: {group_name}")
                sys.exit(1)
            counts = {t: len(v) for t, v in preset_vocabs[group_name].items() if v}
            print(f"  {group_name}: {counts}")
        results = run_full_pipeline(groups, train_cfg, preset_vocabs=preset_vocabs)
    elif "prune_from" in exp:
        # two-phase: train wide model, prune by importance, retrain
        wide_exp = EXPERIMENTS[exp["prune_from"]]
        wide_groups = resolve_groups(exp["prune_from"])
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
                "pentagrams": target.pentagrams,
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

    # phase 3: quantization-aware fine-tuning for product-quantized groups (optional).
    # attaches a fixed PQ codebook to each QAT'd checkpoint; export.py picks that up
    # automatically to produce v4 PQ binaries.
    qat_cfg = exp.get("qat_pq")
    pq_codebooks: dict[str, np.ndarray] = {}
    if qat_cfg:
        qat_groups = qat_cfg.get("groups", [])
        print(f"\n=== phase 3: QAT for product quantization on {qat_groups} ===")
        all_nn_langs = list(dict.fromkeys(
            lang for g in groups.values() for lang in g.langs
        ))
        raw_qat = load_dataset_raw(all_nn_langs, limit=train_cfg.data_limit)
        for group_name in qat_groups:
            if group_name not in results:
                continue
            model_q, vocabs_q, _ = results[group_name]
            tuned, codebook, qat_acc = qat_group_pq(
                groups[group_name], raw_qat, model_q, vocabs_q, train_cfg,
                epochs=qat_cfg.get("epochs", 20),
                lr=qat_cfg.get("lr", 0.001),
                k=qat_cfg.get("k", PQ_K),
                d=qat_cfg.get("d", PQ_D),
                seed=qat_cfg.get("seed", 0),
            )
            results[group_name] = (tuned, vocabs_q, qat_acc)
            pq_codebooks[group_name] = codebook

    total_time = time.time() - t0
    print(f"\ntotal time: {total_time:.1f}s")

    # save checkpoint
    output_name = args.output_name or args.experiment
    ckpt_dir = Path(__file__).parent / "checkpoints" / output_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for group_name, (model, ngram_vocabs, accuracy) in results.items():
        payload: dict[str, object] = {
            "state_dict": model.state_dict(),
            "ngram_vocabs": ngram_vocabs,
            "langs": groups[group_name].langs,
            "accuracy": accuracy,
        }
        if group_name in pq_codebooks:
            payload["pq_codebook"] = pq_codebooks[group_name]
            payload["pq_k"] = PQ_K
            payload["pq_d"] = PQ_D
        torch.save(payload, ckpt_dir / f"{group_name}.pt")
    print(f"checkpoint saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()

"""
product-quantization utilities shared between training (QAT) and export.

PQ replaces each weight row's subvectors (length D) with their nearest centroid
from a K-entry codebook. for linear models this cuts the weight-matrix payload
by roughly 8/D bits per weight at the cost of a small reconstruction error that
QAT can absorb.
"""

from __future__ import annotations

import numpy as np
import torch

# defaults chosen for lean_v5 latin: 256 8-bit indices × 4-dim subvectors.
# see experiment-history for the accuracy/size tradeoff study.
PQ_K = 256
PQ_D = 4


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread initial centroids by D^2-proportional sampling."""
    n = x.shape[0]
    first = int(rng.integers(0, n))
    centers = [x[first]]
    # running min squared distance to any chosen center
    c0 = centers[0]
    d2_min = ((x - c0) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = float(d2_min.sum())
        if total <= 0.0:
            # degenerate: all points already coincide with a center → random pick
            idx = int(rng.integers(0, n))
        else:
            probs = d2_min / total
            idx = int(rng.choice(n, p=probs))
        c = x[idx]
        centers.append(c)
        d2_new = ((x - c) ** 2).sum(axis=1)
        d2_min = np.minimum(d2_min, d2_new)
    return np.stack(centers, axis=0).astype(x.dtype)


def _kmeans_numpy(
    x: np.ndarray,
    k: int,
    iters: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """vanilla k-means on a (n, d) matrix. returns (centroids[k,d], labels[n], mse).

    uses k-means++ seeding for deterministic, well-spread initial centroids.
    empty clusters are re-seeded to the point with the largest residual.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n <= k:
        reps = (k + n - 1) // n
        init = np.tile(x, (reps, 1))[:k].copy().astype(x.dtype)
    else:
        init = _kmeans_pp_init(x, k, rng)
    centroids = init

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # assign: squared euclidean via expansion: d2 = ||x||^2 + ||c||^2 - 2 x c^T
        x2 = (x * x).sum(axis=1, keepdims=True)
        c2 = (centroids * centroids).sum(axis=1)
        d2 = x2 + c2 - 2.0 * (x @ centroids.T)
        labels = np.argmin(d2, axis=1)

        # update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(new_centroids, labels, x)
        np.add.at(counts, labels, 1)
        empty = counts == 0
        if np.any(empty):
            # reseed empty clusters from the points with the largest residual error
            residuals = d2[np.arange(n), labels]
            order = np.argsort(-residuals)
            take = 0
            for ci in np.where(empty)[0]:
                new_centroids[ci] = x[order[take]]
                counts[ci] = 1
                take += 1
        nonempty = ~empty
        new_centroids[nonempty] /= counts[nonempty, None]
        centroids = new_centroids

    # final mse
    x2 = (x * x).sum(axis=1, keepdims=True)
    c2 = (centroids * centroids).sum(axis=1)
    d2 = x2 + c2 - 2.0 * (x @ centroids.T)
    labels = np.argmin(d2, axis=1)
    mse = float(d2[np.arange(n), labels].mean())
    return centroids.astype(np.float32), labels.astype(np.uint8), mse


def pq_encode_weights(
    weight: torch.Tensor,
    k: int = PQ_K,
    d: int = PQ_D,
    seed: int = 0,
    restarts: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[float], int, float]:
    """encode a 2D weight matrix with product quantization on D-dim subvectors.

    per-row absmax scaling is applied first so the codebook operates in unit scale.
    the input dimension is padded with zeros so it divides evenly by D.

    @param weight [out, in] weight matrix
    @param k codebook size (must fit in a uint8 → <= 256)
    @param d subvector length
    @param seed k-means random seed
    @param restarts number of k-means restarts (best-mse is kept)
    @returns (codebook[k,d] f32, indices[out, subvectors] u8, row_scales, padded_in, mse)
    """
    w = weight.detach().cpu().float().numpy()
    out_size, in_size = w.shape

    # per-row absmax normalization. scale here is the divisor: row / absmax ∈ [-1, 1].
    # to match the int6 "scale" convention (multiplier) we store scale = 1/absmax.
    absmax = np.maximum(np.abs(w).max(axis=1), 1e-12)
    row_scale = (1.0 / absmax).astype(np.float32)
    w_norm = w * row_scale[:, None]  # each row now in [-1, 1]

    # pad input dim to a multiple of d
    pad = (-in_size) % d
    padded_in = in_size + pad
    if pad:
        w_norm = np.concatenate([w_norm, np.zeros((out_size, pad), dtype=w_norm.dtype)], axis=1)

    # reshape to (out * subvectors, d) for k-means training over all subvectors
    subvectors = padded_in // d
    x = w_norm.reshape(out_size * subvectors, d)
    # run multiple restarts with different seeds, keep the lowest-mse clustering
    best_mse = float("inf")
    best_centroids = None
    best_labels = None
    for s in range(restarts):
        centroids_s, labels_s, mse_s = _kmeans_numpy(x, k=k, iters=100, seed=seed + s)
        if mse_s < best_mse:
            best_mse = mse_s
            best_centroids = centroids_s
            best_labels = labels_s
    assert best_centroids is not None and best_labels is not None
    indices = best_labels.reshape(out_size, subvectors)

    return best_centroids, indices, row_scale.tolist(), padded_in, best_mse


def pq_assign_indices(
    weight: torch.Tensor,
    centroids: np.ndarray,
    d: int = PQ_D,
) -> tuple[np.ndarray, np.ndarray, list[float], int, float]:
    """encode a 2D weight matrix using a pre-existing PQ codebook (no k-means).

    identical normalization/padding as pq_encode_weights — only codebook training
    is skipped. used when weights were QAT-tuned against a specific fixed codebook.
    """
    w = weight.detach().cpu().float().numpy()
    out_size, in_size = w.shape

    absmax = np.maximum(np.abs(w).max(axis=1), 1e-12)
    row_scale = (1.0 / absmax).astype(np.float32)
    w_norm = w * row_scale[:, None]

    pad = (-in_size) % d
    padded_in = in_size + pad
    if pad:
        w_norm = np.concatenate([w_norm, np.zeros((out_size, pad), dtype=w_norm.dtype)], axis=1)

    subvectors = padded_in // d
    x = w_norm.reshape(out_size * subvectors, d)

    centroids32 = centroids.astype(np.float32)
    x2 = (x * x).sum(axis=1, keepdims=True)
    c2 = (centroids32 * centroids32).sum(axis=1)
    d2 = x2 + c2 - 2.0 * (x @ centroids32.T)
    labels = np.argmin(d2, axis=1).astype(np.uint8)
    mse = float(d2[np.arange(x.shape[0]), labels].mean())
    indices = labels.reshape(out_size, subvectors)

    return centroids32, indices, row_scale.tolist(), padded_in, mse


def pq_snap_weights(
    weight: torch.Tensor,
    codebook: torch.Tensor,
    d: int = PQ_D,
) -> torch.Tensor:
    """snap a weight matrix's subvectors to their nearest codebook centroid.

    differentiable via the caller's straight-through estimator. per-row absmax
    normalization is applied so subvectors live in the codebook's scale. the
    returned tensor has the same shape as the input.

    @param weight (out, in) float tensor — the current weights
    @param codebook (k, d) float tensor — fixed centroids
    @param d subvector dimension
    @returns (out, in) PQ-reconstructed weight tensor
    """
    out_size, in_size = weight.shape
    pad = (-in_size) % d
    padded_in = in_size + pad

    # per-row absmax with clamp to avoid divide-by-zero
    absmax = weight.detach().abs().amax(dim=1).clamp_min(1e-12)
    w_norm = weight / absmax.unsqueeze(1)

    if pad:
        w_norm = torch.cat(
            [w_norm, torch.zeros(out_size, pad, dtype=w_norm.dtype, device=w_norm.device)],
            dim=1,
        )

    # (out, subvectors, d)
    subvectors = padded_in // d
    sv = w_norm.reshape(out_size, subvectors, d)
    flat_sv = sv.reshape(out_size * subvectors, d)

    # nearest centroid by squared euclidean distance
    x2 = (flat_sv * flat_sv).sum(dim=1, keepdim=True)
    c2 = (codebook * codebook).sum(dim=1)
    d2 = x2 + c2 - 2.0 * (flat_sv @ codebook.T)
    labels = torch.argmin(d2, dim=1)

    # gather centroids and reshape back
    snapped = codebook[labels].reshape(out_size, subvectors, d)
    snapped = snapped.reshape(out_size, padded_in)[:, :in_size]

    # denormalize
    return snapped * absmax.unsqueeze(1)

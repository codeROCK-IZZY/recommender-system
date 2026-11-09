"""Splits, top-K ranking and the metrics every model is judged by."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .data import Dataset


@dataclass(frozen=True)
class Split:
    train: sp.csr_matrix  # user x item counts
    train_days: sp.csr_matrix  # user x item, last day seen (for recency)
    test: dict[int, set[int]]  # user -> items first interacted with in the test window
    name: str


def matrix(d: Dataset, mask: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    shape = (d.n_users, d.n_items)
    u, i, day = d.users[mask], d.items[mask], d.days[mask]
    counts = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=shape)  # duplicates add up to counts
    # the latest day per (user, item): sort by day, keep the last occurrence of each pair
    order = np.argsort(day, kind="stable")
    key = u[order].astype(np.int64) * shape[1] + i[order]
    _, last = np.unique(key[::-1], return_index=True)
    pick = order[::-1][last]
    days = sp.csr_matrix((day[pick].astype(float) + 1.0, (u[pick], i[pick])), shape=shape)
    return counts.tocsr(), days


def temporal_split(d: Dataset, cut: int, horizon: int = 30) -> Split:
    """Train on everything before day `cut`; test on items each user first touches in the next `horizon` days."""
    train_mask = d.days < cut
    train, days = matrix(d, train_mask)
    test: dict[int, set[int]] = {}
    tm = (d.days >= cut) & (d.days < cut + horizon)
    for u, i in zip(d.users[tm], d.items[tm], strict=True):
        if train[u, i] == 0:
            test.setdefault(int(u), set()).add(int(i))
    return Split(train, days, test, f"temporal (train < day {cut}, test {cut}-{cut + horizon})")


def random_split(d: Dataset, frac: float = 0.2, seed: int = 0) -> Split:
    """The common shortcut: hold out random interactions, so the model trains on the future."""
    rng = np.random.default_rng(seed)
    pairs = np.unique(np.stack([d.users, d.items], axis=1), axis=0)
    held = rng.random(len(pairs)) < frac
    keep = {(int(u), int(i)) for u, i in pairs[~held]}
    mask = np.array([(int(u), int(i)) in keep for u, i in zip(d.users, d.items, strict=True)])
    train, days = matrix(d, mask)
    test: dict[int, set[int]] = {}
    for u, i in pairs[held]:
        test.setdefault(int(u), set()).add(int(i))
    return Split(train, days, test, "random 80/20 split")


def top_k(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    s = scores.astype(float, copy=True)
    s[seen.nonzero()] = -np.inf
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    return idx[rows, np.argsort(-s[rows, idx], axis=1)]


def recall_ndcg(recs: np.ndarray, truth: list[set[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    rec = np.zeros(len(truth))
    ndcg = np.zeros(len(truth))
    for n, (r, t) in enumerate(zip(recs, truth, strict=True)):
        hit = np.fromiter((i in t for i in r[:k]), bool, k)
        rec[n] = hit.sum() / min(k, len(t))
        ideal = discount[: min(k, len(t))].sum()
        ndcg[n] = (hit * discount).sum() / ideal
    return rec, ndcg


def bootstrap_diff(a: np.ndarray, b: np.ndarray, b_n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(b_n, len(d)))
    m = d[idx].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)

"""Recommenders, from the baseline every model must beat to matrix factorisation.

All of them fit on a sparse user x item count matrix and score a batch of users; items a user has
already interacted with are removed by the caller.

  popularity       most interactions overall
  recent popular   most interactions in the last two weeks of training
  item-kNN         cosine similarity between items' audiences, top neighbours kept
  EASE             closed-form linear autoencoder (Steck, 2019): B = I - P / diag(P), P = (XᵀX + λI)⁻¹
  ALS              implicit-feedback matrix factorisation with confidence 1 + α·count (Hu, Koren, Volinsky)
  BPR              pairwise ranking loss, mini-batch SGD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp


class Model(Protocol):
    name: str

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> Model: ...

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray: ...


@dataclass
class Popularity:
    name: str = "popularity"
    recent_days: int | None = None
    pop: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> Popularity:
        if self.recent_days is not None and days is not None:
            last = days.data.max()
            recent = (days.data > last - self.recent_days).astype(float)
            m = sp.csr_matrix((recent, days.indices, days.indptr), shape=days.shape)
            self.pop = np.asarray(m.sum(axis=0)).ravel() + 1e-3 * np.asarray((x > 0).sum(axis=0)).ravel()
        else:
            self.pop = np.asarray((x > 0).sum(axis=0)).ravel().astype(float)
        return self

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.pop, (len(users), 1))


@dataclass
class ItemKNN:
    name: str = "item-kNN"
    k: int = 100
    sim: sp.csr_matrix | None = None

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> ItemKNN:
        b = (x > 0).astype(float).tocsc()
        norms = np.sqrt(np.asarray(b.multiply(b).sum(axis=0)).ravel()) + 1e-9
        s = (b.T @ b).toarray() / np.outer(norms, norms)
        np.fill_diagonal(s, 0.0)
        if self.k < s.shape[0]:
            cut = np.partition(s, -self.k, axis=1)[:, -self.k][:, None]
            s[s < cut] = 0.0
        self.sim = sp.csr_matrix(s)
        return self

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
        assert self.sim is not None
        return np.asarray(((x[users] > 0).astype(float) @ self.sim).todense())


@dataclass
class EASE:
    name: str = "EASE"
    lam: float = 200.0
    b: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> EASE:
        xb = (x > 0).astype(float)
        g = (xb.T @ xb).toarray()
        g[np.diag_indices_from(g)] += self.lam
        p = np.linalg.inv(g)
        self.b = -p / np.diag(p)
        np.fill_diagonal(self.b, 0.0)
        return self

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
        return np.asarray((x[users] > 0).astype(float) @ self.b)


@dataclass
class ALS:
    name: str = "ALS"
    factors: int = 64
    reg: float = 0.05
    alpha: float = 10.0
    iters: int = 12
    seed: int = 0
    u: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @staticmethod
    def _half(c: sp.csr_matrix, fixed: np.ndarray, reg: float) -> np.ndarray:
        f = fixed.shape[1]
        gram = fixed.T @ fixed + reg * np.eye(f)
        out = np.zeros((c.shape[0], f))
        for row in range(c.shape[0]):
            lo, hi = c.indptr[row], c.indptr[row + 1]
            if lo == hi:
                continue
            idx, conf = c.indices[lo:hi], c.data[lo:hi]
            y = fixed[idx]
            a = gram + (y.T * (conf - 1.0)) @ y
            out[row] = np.linalg.solve(a, (y.T * conf).sum(axis=1))
        return out

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> ALS:
        rng = np.random.default_rng(self.seed)
        c = x.astype(float).tocsr()
        c.data = 1.0 + self.alpha * np.log1p(c.data)
        ct = c.T.tocsr()
        self.v = 0.01 * rng.normal(size=(x.shape[1], self.factors))
        for _ in range(self.iters):
            self.u = self._half(c, self.v, self.reg)
            self.v = self._half(ct, self.u, self.reg)
        return self

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
        return np.asarray(self.u[users] @ self.v.T)


@dataclass
class BPR:
    name: str = "BPR"
    factors: int = 64
    lr: float = 0.05
    reg: float = 0.002
    epochs: int = 40
    batch: int = 4096
    seed: int = 0
    u: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> BPR:
        rng = np.random.default_rng(self.seed)
        coo = (x > 0).tocoo()
        pu, pi = coo.row, coo.col
        n_u, n_i = x.shape
        self.u = 0.1 * rng.normal(size=(n_u, self.factors))
        self.v = 0.1 * rng.normal(size=(n_i, self.factors))
        self.bias = np.zeros(n_i)
        seen = x.tocsr()
        for _ in range(self.epochs):
            order = rng.permutation(len(pu))
            for s in range(0, len(order), self.batch):
                b = order[s : s + self.batch]
                u, i = pu[b], pi[b]
                j = rng.integers(0, n_i, len(b))
                ok = np.asarray(seen[u, j]).ravel() == 0  # negatives the user hasn't seen
                u, i, j = u[ok], i[ok], j[ok]
                wu, hi, hj = self.u[u], self.v[i], self.v[j]
                xuij = (wu * (hi - hj)).sum(axis=1) + self.bias[i] - self.bias[j]
                g = 1.0 / (1.0 + np.exp(np.clip(xuij, -30, 30)))  # d/dx log sigmoid(x)
                du = g[:, None] * (hi - hj) - self.reg * wu
                di = g[:, None] * wu - self.reg * hi
                dj = -g[:, None] * wu - self.reg * hj
                np.add.at(self.u, u, self.lr * du)
                np.add.at(self.v, i, self.lr * di)
                np.add.at(self.v, j, self.lr * dj)
                np.add.at(self.bias, i, self.lr * (g - self.reg * self.bias[i]))
                np.add.at(self.bias, j, self.lr * (-g - self.reg * self.bias[j]))
        return self

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
        return np.asarray(self.u[users] @ self.v.T + self.bias)


@dataclass
class Hybrid:
    """For users with little history: blend a collaborative model with genre affinity and popularity.

    A user's genre profile comes from whatever few items they did touch; with none at all, popularity
    is all there is. The blend weight shifts towards the collaborative model as history grows.
    """

    base: Model
    item_features: np.ndarray
    pop: Popularity = field(default_factory=Popularity)
    name: str = "hybrid"
    full_at: int = 8

    def fit(self, x: sp.csr_matrix, days: sp.csr_matrix | None = None) -> Hybrid:
        self.base.fit(x, days)
        self.pop.fit(x, days)
        self.name = f"{self.base.name} + content fallback"
        return self

    def score(self, users: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
        def z(m: np.ndarray) -> np.ndarray:
            return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

        xb = (x[users] > 0).astype(float)
        n = np.asarray(xb.sum(axis=1)).ravel()
        prof = np.asarray(xb @ self.item_features)
        prof = prof / (prof.sum(axis=1, keepdims=True) + 1e-9)
        content = prof @ self.item_features.T + 0.15 * np.log1p(self.pop.pop)[None, :] / np.log1p(self.pop.pop.max())
        w = np.minimum(1.0, n / self.full_at)[:, None]
        cf = z(self.base.score(users, x))
        out: np.ndarray = w * cf + (1 - w) * z(content)
        return out

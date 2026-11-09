"""Serving: one user's top-K, items like an item, and an HTTP API around them."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from . import models


class Recommender:
    def __init__(self, model: models.Model, interactions: sp.csr_matrix) -> None:
        self.model = model
        self.x = interactions.tocsr()
        self.popular = np.argsort(-np.asarray((self.x > 0).sum(axis=0)).ravel())

    def recommend(self, user: int, k: int = 10) -> list[int]:
        if not 0 <= user < self.x.shape[0]:
            return [int(i) for i in self.popular[:k]]  # an unknown user gets the most popular items
        s = self.model.score(np.array([user]), self.x)[0].astype(float)
        s[self.x[user].indices] = -np.inf
        top = np.argpartition(-s, k)[:k]
        return [int(i) for i in top[np.argsort(-s[top])]]

    def similar(self, item: int, k: int = 10) -> list[int]:
        col = (self.x[:, item] > 0).astype(float)
        co = np.asarray(((self.x > 0).astype(float).T @ col).todense()).ravel()
        norms = np.sqrt(np.asarray((self.x > 0).sum(axis=0)).ravel()) * np.sqrt(max(col.sum(), 1.0)) + 1e-9
        sim = co / norms
        sim[item] = -np.inf
        top = np.argpartition(-sim, k)[:k]
        return [int(i) for i in top[np.argsort(-sim[top])]]


def create_app(service: Recommender) -> Any:
    from fastapi import FastAPI, HTTPException, Query

    app = FastAPI(title="Recommender", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/users/{user}/recommendations")
    def recommendations(user: int, k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        return {"user": user, "items": service.recommend(user, k)}

    @app.get("/items/{item}/similar")
    def similar(item: int, k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        if not 0 <= item < service.x.shape[1]:
            raise HTTPException(404, "unknown item")
        return {"item": item, "items": service.similar(item, k)}

    return app

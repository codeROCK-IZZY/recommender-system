"""The evaluation: tune on a validation window, report on a later test window, never the other way round."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import data, metrics, models
from .serve import Recommender

K = 20
VAL_CUT, TEST_CUT, HORIZON = 120, 150, 30

Factory = Callable[[dict[str, Any]], models.Model]


def grids(item_features: np.ndarray, quick: bool) -> dict[str, tuple[Factory, list[dict[str, Any]]]]:
    als = [{"factors": 64, "reg": r, "alpha": a, "iters": 8} for r, a in ((300.0, 40.0), (30.0, 20.0), (100.0, 20.0))]
    bpr = [{"factors": 64, "lr": lr, "epochs": 15 if quick else 30} for lr in (0.03, 0.08)]
    return {
        "popularity": (lambda p: models.Popularity(), [{}]),
        "recent popular": (
            lambda p: models.Popularity("recent popular", **p),
            [{"recent_days": 7}, {"recent_days": 21}],
        ),
        "item-kNN": (lambda p: models.ItemKNN(**p), [{"k": 50}, {"k": 200}]),
        "EASE": (lambda p: models.EASE(**p), [{"lam": lam} for lam in (250.0, 1000.0, 3000.0)]),
        "ALS": (lambda p: models.ALS(**p), als[:1] if quick else als),
        "BPR": (lambda p: models.BPR(**p), bpr[:1] if quick else bpr),
        "EASE + content fallback": (
            lambda p: models.Hybrid(models.EASE(lam=p["lam"]), item_features, full_at=p["full_at"]),
            [{"lam": 1000.0, "full_at": f} for f in (4, 8, 16)],
        ),
    }


@dataclass
class Result:
    seed: int
    interactions: int
    users: int
    items: int
    test_users: int
    cold_users: int
    unseen_item_share: float
    chosen: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    test: dict[str, dict[str, float]] = field(default_factory=dict)
    best: str = ""
    lift_ci: tuple[float, float] = (0.0, 0.0)
    split_inflation: dict[str, float] = field(default_factory=dict)
    serving_ms: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def rank_all(m: models.Model, s: metrics.Split, users: np.ndarray, k: int = K) -> np.ndarray:
    return np.vstack(
        [
            metrics.top_k(m.score(users[a : a + 1000], s.train), s.train[users[a : a + 1000]], k)
            for a in range(0, len(users), 1000)
        ]
    )


def score_split(m: models.Model, s: metrics.Split) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    users = np.array(sorted(s.test))
    recs = rank_all(m, s, users)
    rec, ndcg = metrics.recall_ndcg(recs, [s.test[u] for u in users], K)
    return users, recs, rec, ndcg


def run(seed: int = 7, quick: bool = False) -> Result:
    t0 = time.perf_counter()
    d = data.generate(seed)
    val = metrics.temporal_split(d, VAL_CUT, HORIZON)
    test = metrics.temporal_split(d, TEST_CUT, HORIZON)
    n_train = np.asarray((test.train > 0).sum(axis=1)).ravel()
    pop = np.asarray((test.train > 0).sum(axis=0)).ravel()
    pop_rank = pop.argsort().argsort() / (len(pop) - 1)  # popularity percentile per item
    seen_items = pop > 0
    test_pairs = [(u, i) for u, items in test.test.items() for i in items]
    r = Result(
        seed,
        len(d.users),
        d.n_users,
        d.n_items,
        len(test.test),
        int((n_train[list(test.test)] <= 3).sum()),
        float(np.mean([not seen_items[i] for _, i in test_pairs])),
    )

    per_user: dict[str, np.ndarray] = {}
    fitted: dict[str, models.Model] = {}
    for name, (make, grid) in grids(d.item_features(), quick).items():
        scores = []
        for params in grid:
            m = make(params).fit(val.train, val.train_days)
            scores.append((json.dumps(params), float(score_split(m, val)[3].mean())))
        best = grid[int(np.argmax([s for _, s in scores]))]
        r.validation[name] = scores
        r.chosen[name] = best
        m = make(best).fit(test.train, test.train_days)
        fitted[name] = m
        users, recs, rec, ndcg = score_split(m, test)
        cold = n_train[users] <= 3
        heavy = n_train[users] >= 50
        per_user[name] = ndcg
        r.test[name] = {
            "recall": float(rec.mean()),
            "ndcg": float(ndcg.mean()),
            "ndcg_cold": float(ndcg[cold].mean()),
            "ndcg_heavy": float(ndcg[heavy].mean()),
            "coverage": float(len(np.unique(recs)) / d.n_items),
            "popularity_percentile": float(pop_rank[recs].mean()),
        }
    r.best = max(r.test, key=lambda n: r.test[n]["ndcg"])
    r.lift_ci = metrics.bootstrap_diff(per_user[r.best], per_user["popularity"])

    rnd = metrics.random_split(d)
    ease = models.EASE(lam=r.chosen["EASE"]["lam"])
    r.split_inflation = {
        "random split": float(score_split(ease.fit(rnd.train), rnd)[3].mean()),
        "temporal split": r.test["EASE"]["ndcg"],
        "random split, popularity": float(score_split(models.Popularity().fit(rnd.train), rnd)[3].mean()),
        "temporal split, popularity": r.test["popularity"]["ndcg"],
    }

    service = Recommender(fitted[r.best], test.train)
    times = []
    for u in list(test.test)[:600]:
        t = time.perf_counter()
        service.recommend(int(u), 10)
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times[50:])
    r.serving_ms = {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }
    r.seconds = time.perf_counter() - t0
    return r


def check(r: Result) -> list[str]:
    p = []
    lo, _ = r.lift_ci
    if lo <= 0:
        p.append(f"best model {r.best} does not beat popularity with confidence: CI {r.lift_ci}")
    t = r.test
    if t["EASE + content fallback"]["ndcg_cold"] < t["EASE"]["ndcg_cold"] * 1.5:
        p.append("the content fallback should rescue cold-start users")
    if r.split_inflation["random split"] <= r.split_inflation["temporal split"]:
        p.append("a random split should look better than a temporal one; the comparison is broken")
    if t[r.best]["coverage"] <= t["popularity"]["coverage"]:
        p.append("the best model should cover more of the catalogue than popularity")
    if r.serving_ms["p95"] > 25:
        p.append(f"serving p95 {r.serving_ms['p95']:.1f} ms > 25 ms")
    return p


def to_json(r: Result) -> str:
    return json.dumps(asdict(r), indent=2) + "\n"


def to_markdown(r: Result) -> str:
    L = [
        "# Evaluation",
        "",
        f"Seed {r.seed}: {r.interactions:,} interactions, {r.users:,} users, {r.items:,} items. Hyperparameters chosen on "
        f"a validation window (train before day {VAL_CUT}, test days {VAL_CUT}-{VAL_CUT + HORIZON}); the numbers below "
        f"are from a later test window (train before day {TEST_CUT}, test days {TEST_CUT}-{TEST_CUT + HORIZON}), "
        f"{r.test_users:,} users, of whom {r.cold_users} are cold (3 or fewer interactions). Generated by `recsys evaluate`.",
        "",
        f"## Test window, top {K}",
        "",
        "| model | NDCG | recall | NDCG, cold users | NDCG, heavy users | catalogue coverage | avg popularity percentile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in r.test.items():
        star = "**" if name == r.best else ""
        L.append(
            f"| {star}{name}{star} | {m['ndcg']:.4f} | {m['recall']:.4f} | {m['ndcg_cold']:.4f} | {m['ndcg_heavy']:.4f} | "
            f"{m['coverage']:.1%} | {m['popularity_percentile']:.2f} |"
        )
    pop = r.test["popularity"]["ndcg"]
    L += [
        "",
        f"Best: **{r.best}**, NDCG {r.test[r.best]['ndcg']:.4f} against popularity {pop:.4f} "
        f"(+{(r.test[r.best]['ndcg'] / pop - 1):.0%}); paired bootstrap 95% CI of the per-user difference "
        f"[{r.lift_ci[0]:+.4f}, {r.lift_ci[1]:+.4f}].",
        "",
        f"{r.unseen_item_share:.1%} of test interactions are with items no one touched in training (new releases); "
        "no collaborative model can rank those.",
        "",
        "## Hyperparameters (chosen on validation)",
        "",
        "| model | grid (validation NDCG) | chosen |",
        "|---|---|---|",
    ]
    for name, grid in r.validation.items():
        L.append(
            f"| {name} | " + "<br>".join(f"`{g}` {s:.4f}" for g, s in grid) + f" | `{json.dumps(r.chosen[name])}` |"
        )
    s = r.split_inflation
    L += [
        "",
        "## Why the split matters",
        "",
        "| | EASE NDCG | popularity NDCG |",
        "|---|---:|---:|",
        f"| random 80/20 split | {s['random split']:.4f} | {s['random split, popularity']:.4f} |",
        f"| temporal split | {s['temporal split']:.4f} | {s['temporal split, popularity']:.4f} |",
        "",
        "## Serving",
        "",
        f"One user's top 10 from the best model: p50 {r.serving_ms['p50']:.2f} ms, "
        f"p95 {r.serving_ms['p95']:.2f} ms, p99 {r.serving_ms['p99']:.2f} ms.",
        "",
    ]
    return "\n".join(L)

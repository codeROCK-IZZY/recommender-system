from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from recsys import data, metrics, models
from recsys.serve import Recommender, create_app


@pytest.fixture(scope="module")
def small() -> data.Dataset:
    return data.generate(seed=3, n_users=800, n_items=300, mean_events=30.0)


@pytest.fixture(scope="module")
def split(small: data.Dataset) -> metrics.Split:
    return metrics.temporal_split(small, 150)


def test_generator_respects_release_days_and_signups(small: data.Dataset) -> None:
    assert (small.days >= small.item_release[small.items]).all()
    assert (small.days >= small.user_start[small.users]).all()
    counts = np.bincount(small.items, minlength=small.n_items)
    assert np.sort(counts)[::-1][: small.n_items // 20].sum() > 0.15 * counts.sum()  # a long tail


def test_temporal_split_never_trains_on_the_future(small: data.Dataset, split: metrics.Split) -> None:
    assert split.train_days.data.max() <= 150  # days are stored +1
    for u, items in list(split.test.items())[:200]:
        assert all(split.train[u, i] == 0 for i in items)  # test items are new to the user


def test_latest_day_is_kept_per_pair() -> None:
    d = data.Dataset(
        np.array([0, 0, 0]),
        np.array([1, 1, 2]),
        np.array([5, 9, 3]),
        1,
        3,
        np.zeros(3, int),
        np.zeros(3, int),
        np.zeros(1, int),
        1,
    )
    counts, days = metrics.matrix(d, np.ones(3, bool))
    assert counts[0, 1] == 2
    assert days[0, 1] == 10
    assert days[0, 2] == 4


def test_metrics_on_a_known_ranking() -> None:
    recs = np.array([[1, 2, 3], [4, 5, 6]])
    rec, ndcg = metrics.recall_ndcg(recs, [{1}, {9}], 3)
    assert rec.tolist() == [1.0, 0.0]
    assert ndcg[0] == pytest.approx(1.0)
    seen = sp.csr_matrix(np.array([[0, 1, 0, 0]]))
    assert metrics.top_k(np.array([[0.1, 0.9, 0.5, 0.2]]), seen, 2).tolist() == [[2, 3]]


@pytest.mark.parametrize(
    "model",
    [
        models.Popularity(),
        models.ItemKNN(k=30),
        models.EASE(lam=100.0),
        models.ALS(factors=8, iters=3, reg=10.0),
        models.BPR(factors=8, epochs=3),
    ],
)
def test_every_model_scores_every_item(model: models.Model, split: metrics.Split) -> None:
    model.fit(split.train, split.train_days)
    s = model.score(np.array([0, 1, 2]), split.train)
    assert s.shape == (3, split.train.shape[1])
    assert np.isfinite(s).all()


def test_personalised_models_beat_popularity() -> None:
    # at a toy scale (a few hundred items) popularity genuinely wins; the lift needs a realistic catalogue
    split = metrics.temporal_split(data.generate(seed=3, n_users=2500, n_items=800), 150)
    users = np.array(sorted(split.test))
    truth = [split.test[u] for u in users]

    def ndcg(m: models.Model) -> float:
        m.fit(split.train, split.train_days)
        return float(
            metrics.recall_ndcg(metrics.top_k(m.score(users, split.train), split.train[users], 20), truth, 20)[1].mean()
        )

    assert ndcg(models.EASE(lam=500.0)) > ndcg(models.Popularity())


def test_hybrid_helps_cold_users(small: data.Dataset, split: metrics.Split) -> None:
    hybrid = models.Hybrid(models.EASE(lam=200.0), small.item_features()).fit(split.train, split.train_days)
    one = sp.csr_matrix(([1.0], ([0], [int(np.argmax(small.item_genre == 2))])), shape=split.train.shape)
    s = hybrid.score(np.array([0]), one)[0]
    top = np.argsort(-s)[:20]
    assert (small.item_genre[top] == 2).mean() > 0.5  # one item of genre 2 points the user towards genre 2


def test_serving_api(small: data.Dataset, split: metrics.Split) -> None:
    svc = Recommender(models.EASE(lam=200.0).fit(split.train), split.train)
    recs = svc.recommend(5, 10)
    assert len(recs) == 10
    assert not set(recs) & set(split.train[5].indices)  # never what the user already has
    client = TestClient(create_app(svc))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert len(client.get("/users/5/recommendations?k=5").json()["items"]) == 5
    assert len(client.get("/users/999999/recommendations").json()["items"]) == 10  # unknown user: popular items
    assert client.get("/items/99999/similar").status_code == 404
    assert 7 not in client.get("/items/7/similar").json()["items"]

"""A synthetic shop with the properties that make recommendation hard.

  - a long tail: a few items get most of the attention, whatever people actually like
  - taste: users like a mix of 1 to 3 genres, items sit near their genre in a latent space
  - drift: over the six months, each user's taste moves towards another genre
  - new items: a fifth of the catalogue is released during the period
  - cold start: a tenth of users sign up in the last weeks, with only a few interactions
  - repeat views: implicit feedback is counts, not ratings

Interactions are drawn from a softmax over items of (taste match + log popularity), so the generator
knows nothing about any of the models evaluated on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAYS = 180


@dataclass(frozen=True)
class Dataset:
    users: np.ndarray  # (n,) user id per interaction
    items: np.ndarray  # (n,) item id
    days: np.ndarray  # (n,) day of the interaction, 0..DAYS-1
    n_users: int
    n_items: int
    item_genre: np.ndarray  # (n_items,) genre id
    item_release: np.ndarray  # (n_items,) first day the item can be seen
    user_start: np.ndarray  # (n_users,) sign-up day
    genres: int

    def item_features(self) -> np.ndarray:
        """Content features for cold start: genre one-hot."""
        f = np.zeros((self.n_items, self.genres))
        f[np.arange(self.n_items), self.item_genre] = 1.0
        return f


def generate(
    seed: int = 7, n_users: int = 6000, n_items: int = 2000, genres: int = 12, dim: int = 16, mean_events: float = 45.0
) -> Dataset:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(genres, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    item_genre = rng.integers(0, genres, n_items)
    item_vec = centroids[item_genre] + 0.45 * rng.normal(size=(n_items, dim)) / np.sqrt(dim)
    popularity = rng.zipf(1.6, n_items).astype(float)
    log_pop = np.log(np.minimum(popularity, 400.0))
    item_release = np.where(rng.random(n_items) < 0.2, rng.integers(20, DAYS - 10, n_items), 0)

    user_start = np.where(
        rng.random(n_users) < 0.1, rng.integers(DAYS - 40, DAYS - 5, n_users), rng.integers(0, 30, n_users)
    )
    n_tastes = rng.integers(1, 4, n_users)
    activity = rng.lognormal(0.0, 0.8, n_users)
    activity *= mean_events / activity.mean()

    us, its, ds = [], [], []
    beta = 4.0
    for u in range(n_users):
        g = rng.choice(genres, n_tastes[u], replace=False)
        start_vec = centroids[g].mean(axis=0) + 0.3 * rng.normal(size=dim) / np.sqrt(dim)
        end_vec = 0.6 * start_vec + 0.4 * centroids[rng.integers(genres)]
        span = DAYS - user_start[u]
        n = int(rng.poisson(activity[u] * span / DAYS)) + (2 if user_start[u] > DAYS - 45 else 1)
        days = np.sort(rng.integers(user_start[u], DAYS, n))
        # three phases of the user's life on the site, each with its own taste vector (drift)
        for phase in range(3):
            lo = user_start[u] + span * phase / 3
            hi = user_start[u] + span * (phase + 1) / 3
            sel = days[(days >= lo) & (days < hi)]
            if not len(sel):
                continue
            w = phase / 2
            vec = (1 - w) * start_vec + w * end_vec
            available = item_release <= sel.max()
            logits = beta * item_vec @ vec + 0.9 * log_pop
            logits[~available] = -np.inf
            p = np.exp(logits - logits[available].max())
            p /= p.sum()
            picks = rng.choice(n_items, len(sel), p=p)
            ok = item_release[picks] <= sel  # an item can't be seen before it is released
            us.append(np.full(ok.sum(), u))
            its.append(picks[ok])
            ds.append(sel[ok])
    return Dataset(
        np.concatenate(us),
        np.concatenate(its),
        np.concatenate(ds),
        n_users,
        n_items,
        item_genre,
        item_release,
        user_start,
        genres,
    )

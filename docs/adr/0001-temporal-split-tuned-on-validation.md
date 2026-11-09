# ADR 1: A temporal split, with tuning and reporting on different windows

## Context

The quickest way to evaluate a recommender is to hold out a random 20% of interactions. That lets
the model learn from the future: it sees what a user did next week while predicting what they did
this week, and popular new items are already in its training data. On this data a random split makes
EASE look 65% better than it is. Tuning hyperparameters on the same data the results are reported on
inflates the numbers further.

## Decision

- **Validation window:** train on days 0–119, measure on the items each user first touches on days
  120–149. Every model's hyperparameters are chosen here, from a grid recorded in the report.
- **Test window:** refit on days 0–149 with the chosen settings, then measure once on days 150–179.
  Every reported number comes from this window.
- Test items are those *new to the user*. Re-recommending what someone already has is not counted.
- Differences between models come with a paired bootstrap interval over users.

## Consequences

- The numbers are lower than random-split numbers in papers, and they are the numbers that matter in
  production.
- Each model gets its own grid and the winner is picked on validation, so no model wins because it
  was tuned more. When a model looked weak (ALS at first), the grid was widened, not the model
  dropped.
- New items released in the test window (3.9% of test interactions here) can't be ranked by any
  collaborative model. Fixing that needs content features, which is the fallback's job.

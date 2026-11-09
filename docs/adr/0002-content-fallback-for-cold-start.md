# ADR 2: Blend in content and popularity for users with little history

## Context

Collaborative models learn from co-occurrence. A user with one or two interactions gives them almost
nothing to work with, so on cold-start users EASE and item-kNN score about a third of what plain
popularity does. A tenth of users are new in any given month.

## Decision

Score every user with a blend of z-scored model scores:

- the collaborative model, with weight `min(1, history / full_at)`;
- content affinity (the genre profile of whatever the user did touch) plus a little popularity, with
  the remaining weight.

`full_at` is a hyperparameter chosen on validation like any other. A user with no history at all gets
the most popular items, which is the honest best guess.

## Consequences

- Cold-start NDCG recovers to popularity's level and overall NDCG improves, because the blend also
  helps users with a handful of interactions.
- The content features here are genres only. Richer features (text, price, brand) would help new
  items as well as new users. That's listed in the known limitations.

# Known limitations

- **Synthetic data.** Interactions come from a generator with a long tail, taste drift, new items and
  late sign-ups, but it is not a real shop. The relative findings (popularity is a hard baseline, a
  random split flatters models, cold start needs content) match published experience. The absolute
  NDCG values don't transfer.
- **Offline metrics only.** NDCG on held-out interactions measures how well a model predicts what
  people did without it, not what they would do with it. Online A/B tests are the real judge.
- **New items.** 3.9% of test interactions are with items nobody touched in training. No collaborative
  model can recommend those, and genre is the only content feature.
- **Popularity bias.** Even the best model recommends items from the top few percentiles of
  popularity on average. Re-ranking for diversity or calibration is not implemented.
- **Batch training.** Models are refit from scratch. There is no incremental update, and serving
  scores a user against the whole catalogue (fine at 2,000 items; a larger catalogue would need an
  approximate nearest-neighbour index).
- **BPR** is trained with uniform negative sampling and few epochs. It is the least tuned model here.

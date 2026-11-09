"""A self-contained HTML report of the evaluation."""

from __future__ import annotations

import html

from .evaluate import K, Result

CSS = """
:root { --bg:#0b0e13; --panel:#121720; --line:#232c3a; --ink:#e8ecf2; --ink2:#9aa6b8; --s1:#3987e5; --s2:#d95926;
  font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 40px 28px 64px; }
h1 { font-size: 30px; letter-spacing:-0.02em; margin: 0 0 6px; } h2 { font-size: 19px; margin: 34px 0 14px; }
.sub, .muted { color: var(--ink2); } .sub { margin: 0 0 24px; }
.tiles { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.tile b { display:block; font-size: 28px; letter-spacing:-0.02em; } .tile span { color: var(--ink2); font-size: 13px; }
table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-size: 13.5px; }
th, td { padding: 8px 12px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; } th { color: var(--ink2); font-weight:500; font-size:12px; }
tr.best td { background: rgba(57,135,229,.10); font-weight: 600; }
.bar { display:inline-block; height: 8px; border-radius: 0 4px 4px 0; vertical-align: middle; margin-right: 8px; }
.b1 { background: var(--s1); } .b2 { background: var(--s2); }
.cols { display:grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.legend span { margin-right: 18px; font-size: 13px; color: var(--ink2); }
"""


def render(r: Result) -> str:
    best = r.test[r.best]
    pop = r.test["popularity"]
    tiles = [
        (
            f"+{best['ndcg'] / pop['ndcg'] - 1:.0%}",
            f"NDCG@{K} of {r.best} over popularity (CI of the difference {r.lift_ci[0]:+.3f} to {r.lift_ci[1]:+.3f})",
        ),
        (f"{best['ndcg_cold']:.3f}", f"NDCG for cold-start users (EASE alone: {r.test['EASE']['ndcg_cold']:.3f})"),
        (
            f"{r.split_inflation['random split'] / r.split_inflation['temporal split'] - 1:+.0%}",
            "how much a random split flatters EASE versus a temporal one",
        ),
        (f"{r.serving_ms['p95']:.1f} ms", "p95 to serve one user's top 10"),
    ]
    t = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    top = max(m["ndcg"] for m in r.test.values())
    rows = "".join(
        f'<tr class="{"best" if n == r.best else ""}"><td>{html.escape(n)}</td>'
        f'<td><span class="bar b1" style="width:{int(m["ndcg"] / top * 120)}px"></span>{m["ndcg"]:.4f}</td>'
        f'<td><span class="bar b2" style="width:{int(m["ndcg_cold"] / top * 120)}px"></span>{m["ndcg_cold"]:.4f}</td>'
        f"<td>{m['recall']:.4f}</td><td>{m['coverage']:.1%}</td><td>{m['popularity_percentile']:.2f}</td></tr>"
        for n, m in r.test.items()
    )
    s = r.split_inflation
    split = (
        f"<tr><td>random 80/20 split</td><td>{s['random split']:.4f}</td><td>{s['random split, popularity']:.4f}</td></tr>"
        f"<tr><td>temporal split</td><td>{s['temporal split']:.4f}</td><td>{s['temporal split, popularity']:.4f}</td></tr>"
    )
    chosen = "".join(f"<tr><td>{html.escape(n)}</td><td>{html.escape(str(p))}</td></tr>" for n, p in r.chosen.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Recommender System: evaluation</title><style>{CSS}</style></head><body><main>
<h1>Recommendations, measured honestly</h1>
<p class="sub">{r.interactions:,} interactions from {r.users:,} users on {r.items:,} items. Every model is tuned on a
validation window and scored on a later test window: {r.test_users:,} users, {r.cold_users} of them cold.</p>
<div class="tiles">{t}</div>
<h2>Test window, top {K}</h2>
<p class="legend"><span><i class="bar b1" style="width:14px"></i>NDCG, all users</span>
<span><i class="bar b2" style="width:14px"></i>NDCG, cold-start users</span></p>
<table><tr><th>model</th><th>NDCG</th><th>NDCG, cold</th><th>recall</th><th>catalogue coverage</th>
<th>avg popularity percentile</th></tr>{rows}</table>
<div class="cols"><div><h2>Why the split matters</h2>
<table><tr><th></th><th>EASE NDCG</th><th>popularity NDCG</th></tr>{split}</table>
<p class="muted">A random split lets the model train on the future; the temporal split is what it will face.</p></div>
<div><h2>Chosen on validation</h2><table><tr><th>model</th><th>hyperparameters</th></tr>{chosen}</table></div></div>
<p class="muted">{r.unseen_item_share:.1%} of test interactions are with items nobody touched in training. Serving p50
{r.serving_ms["p50"]:.2f} ms, p99 {r.serving_ms["p99"]:.2f} ms.</p>
</main></body></html>
"""

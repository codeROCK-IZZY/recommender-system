"""`recsys`: run the evaluation, recommend for a user, or serve the HTTP API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _service(seed: int) -> object:
    from . import data, metrics, models
    from .serve import Recommender

    d = data.generate(seed)
    s = metrics.temporal_split(d, data.DAYS)  # train on everything
    return Recommender(models.Hybrid(models.EASE(lam=1000.0), d.item_features()).fit(s.train, s.train_days), s.train)


def cmd_recommend(args: argparse.Namespace) -> int:
    svc = _service(args.seed)
    print("user", args.user, "->", svc.recommend(args.user, args.k))  # type: ignore[attr-defined]
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serve import Recommender, create_app

    svc = _service(args.seed)
    assert isinstance(svc, Recommender)
    uvicorn.run(create_app(svc), host=args.host, port=args.port)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import check, run, to_json, to_markdown
    from .report import render

    r = run(seed=args.seed, quick=args.quick)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(to_json(r), newline="\n", encoding="utf-8")
    (out / "evaluation.md").write_text(to_markdown(r), newline="\n", encoding="utf-8")
    (out / "report.html").write_text(render(r), newline="\n", encoding="utf-8")
    print(f"best {r.best} ({r.seconds:.0f}s) -> {out}")
    if args.gate:
        problems = check(r)
        for p in problems:
            print(f"GATE FAILED: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recsys", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="tune on validation, report on test")
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--quick", action="store_true", help="smaller hyperparameter grids (CI)")
    e.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    e.add_argument("--gate", action="store_true")
    e.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("recommend", help="top-K for one user")
    r.add_argument("user", type=int)
    r.add_argument("-k", type=int, default=10)
    r.add_argument("--seed", type=int, default=7)
    r.set_defaults(fn=cmd_recommend)
    s = sub.add_parser("serve", help="the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

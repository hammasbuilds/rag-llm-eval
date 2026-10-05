"""09 - Testing project 03's excuse.

Project 03 found that a second retrieval hop is worth +7.1 points on bridge
questions and costs 11.7 on comparison questions, so the gain depends entirely on
routing each question to the right strategy. Its rule-based router had recall
0.977 and precision **0.386** -- it fired on 477 questions that did not want
correcting -- and the project blamed that for the gap between routed bridge
accuracy (0.820) and always-two-hop (0.840).

That is a convenient excuse, and it is testable: build a better router and see
whether the gap closes. If it does, the claim was right and routing is a
solvable problem. If a much more accurate router buys nothing, the excuse was
wrong and something else is limiting the result.

The router is logistic regression over hand-written features, fitted here in
thirty lines rather than imported, and -- the part that matters -- **fitted on a
disjoint slice of the queries**. Scoring a router on the questions it was tuned
on would reproduce exactly the kind of inflated number this lab exists to avoid.

    train   the first 500 queries
    test    the remaining 1000, never seen during fitting
    report  both routers on the test slice only

    python run.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.corpus import load  # noqa: E402
from shared.metrics import evaluate  # noqa: E402
from shared.pipeline import Index  # noqa: E402
from shared.retrieval import rrf, tokenize  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 1500
N_TRAIN = 500
DEPTH = 20
POOL = 50

_CAPS = re.compile(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*")
_COMPARATIVE = re.compile(
    r"\b(\w+er|more|less|most|least|first|last|older|younger|earlier|later)\b"
)


def features(question: str) -> list[float]:
    """Hand-written features. Each one is a guess about what a comparison looks like."""
    q = question.lower()
    body = question[1:]  # skip the sentence-initial capital
    caps = _CAPS.findall(body)
    return [
        1.0,  # bias
        float("both" in q),
        float("same" in q),
        float(" or " in q),
        float(" and " in q),
        float(len(caps) >= 2),
        float(len(caps) >= 3),
        float(bool(_COMPARATIVE.search(q))),
        float(q.startswith(("which", "who", "what"))),
        float("which came" in q or "came first" in q),
        float(" than " in q),
        float("are " in q[:12] or "were " in q[:12] or "is " in q[:8] or "was " in q[:8]),
        min(len(tokenize(question)), 30) / 30.0,
    ]


def fit_logistic(rows: list[list[float]], labels: list[int], *, epochs: int = 400, lr: float = 0.5):
    """Plain batch gradient descent. No sklearn, no regularisation worth the name."""
    n_features = len(rows[0])
    w = [0.0] * n_features
    n = len(rows)
    for _ in range(epochs):
        grad = [0.0] * n_features
        for x, y in zip(rows, labels, strict=False):
            z = sum(wi * xi for wi, xi in zip(w, x, strict=False))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - y
            for j in range(n_features):
                grad[j] += err * x[j]
        for j in range(n_features):
            w[j] -= lr * grad[j] / n
    return w


def predict(w: list[float], x: list[float], threshold: float = 0.5) -> bool:
    z = sum(wi * xi for wi, xi in zip(w, x, strict=False))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) >= threshold


def score_router(predictions: list[bool], truth: list[bool]) -> dict:
    tp = sum(p and t for p, t in zip(predictions, truth, strict=False))
    fp = sum(p and not t for p, t in zip(predictions, truth, strict=False))
    fn = sum((not p) and t for p, t in zip(predictions, truth, strict=False))
    tn = sum((not p) and (not t) for p, t in zip(predictions, truth, strict=False))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "accuracy": round((tp + tn) / len(truth), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("p03", ROOT / "projects" / "03_iterative_multihop" / "run.py")
    p03 = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(p03)

    corpus, queries = load(N_QUESTIONS)
    train, test = queries[:N_TRAIN], queries[N_TRAIN:]
    index = Index.build(corpus)
    print(f"corpus: {len(corpus)} passages   train: {len(train)}   test: {len(test)}\n")

    # --- fit on train only -------------------------------------------------------
    X = [features(q.question) for q in train]
    y = [1 if q.hop_type == "comparison" else 0 for q in train]
    t0 = time.time()
    w = fit_logistic(X, y)
    print(f"router fitted on {len(train)} held-out queries in {time.time() - t0:.1f}s\n")

    truth = [q.hop_type == "comparison" for q in test]
    rule_pred = [p03.looks_like_comparison(q.question) for q in test]
    learned_pred = [predict(w, features(q.question)) for q in test]

    routers = {
        "rule-based (project 03)": score_router(rule_pred, truth),
        "learned (this project)": score_router(learned_pred, truth),
    }
    for name, s in routers.items():
        print(
            f"{name:26} precision={s['precision']:.3f}  recall={s['recall']:.3f}  "
            f"f1={s['f1']:.3f}  accuracy={s['accuracy']:.3f}"
        )
    print()

    # --- does a better router actually retrieve better? --------------------------
    def retrieve(query: str, k: int):
        return rrf([index.bm25.search(query, POOL), index.dense.search(query, POOL)], k=k)  # type: ignore[union-attr]

    def single(q):
        return [d for d, _ in retrieve(q.question, DEPTH)]

    def two_hop(q):
        first = [d for d, _ in retrieve(q.question, DEPTH)]
        seeds = first[:2]
        q2 = p03.hop2_query(q.question, [corpus.get(d).text for d in seeds])
        second = [d for d, _ in retrieve(q2, DEPTH)]
        merged = list(seeds)
        for d in second + first:
            if d not in merged:
                merged.append(d)
        return merged[:DEPTH]

    strategies = {
        "single hop": lambda q, i: single(q),
        "two hops (always)": lambda q, i: two_hop(q),
        "routed: rule-based": lambda q, i: single(q) if rule_pred[i] else two_hop(q),
        "routed: learned": lambda q, i: single(q) if learned_pred[i] else two_hop(q),
        "routed: ORACLE": lambda q, i: single(q) if truth[i] else two_hop(q),
    }

    results = {"routers": routers, "retrieval": {}}
    for name, fn in strategies.items():
        t0 = time.time()
        ranked = [(fn(q, i), q.gold_doc_ids) for i, q in enumerate(test)]
        overall = evaluate(ranked, seconds=time.time() - t0).as_dict()
        by_type = {}
        for hop_type in ("bridge", "comparison"):
            subset = [
                (r, g) for (r, g), q in zip(ranked, test, strict=False) if q.hop_type == hop_type
            ]
            by_type[hop_type] = evaluate(subset).as_dict()
        results["retrieval"][name] = {"overall": overall, "by_type": by_type}
        print(
            f"{name:22} answerable@10: overall={overall['answerable'][10]:.3f}  "
            f"bridge={by_type['bridge']['answerable'][10]:.3f}  "
            f"comparison={by_type['comparison']['answerable'][10]:.3f}"
        )

    out = HERE / "results.json"
    out.write_text(
        json.dumps(
            {"n_train": len(train), "n_test": len(test), "depth": DEPTH, **results}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()

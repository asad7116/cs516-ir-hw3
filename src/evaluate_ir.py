from __future__ import annotations

from typing import Dict, Set, List

from src.ir_system import (BM25IRSystem, calculate_precision_recall, average_precision, reciprocal_rank)


def main() -> None:
    system = BM25IRSystem()

    # 1) Define a small set of test queries and relevance sets
    #    Use actual doc indices from your own runs.
    QUERIES: List[str] = [
        "oil price cash",
        "stock market pakistan",
        "cricket match today",
    ]

    RELEVANCE: Dict[str, Set[int]] = {
        "oil price cash": {444, 575},       # example – adjust based on your dataset
        "stock market pakistan": {120, 305},
        "cricket match today": {50, 78, 90},
    }

    top_k = 10

    print(f"Evaluating BM25 on {len(QUERIES)} queries (top_k={top_k})\n")
    print("{:<25} {:>9} {:>9} {:>9} {:>9}".format(
        "Query", "Prec", "Recall", "AP", "RR"
    ))
    print("-" * 65)

    precs, recs, aps, rrs = [], [], [], []

    for q in QUERIES:
        rel = RELEVANCE.get(q, set())
        out = system.search(q, top_k=top_k, relevant_docs=rel)

        idxs = out["indices"]

        p, r = calculate_precision_recall(idxs, rel)
        ap = average_precision(idxs, rel)
        rr = reciprocal_rank(idxs, rel)

        precs.append(p)
        recs.append(r)
        aps.append(ap)
        rrs.append(rr)

        print("{:<25} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            q[:24], p, r, ap, rr
        ))

    # macro averages
    if QUERIES:
        avg_p = sum(precs) / len(precs)
        avg_r = sum(recs) / len(recs)
        mean_ap = sum(aps) / len(aps)
        mean_rr = sum(rrs) / len(rrs)

        print("-" * 65)
        print("{:<25} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            "MEAN", avg_p, avg_r, mean_ap, mean_rr
        ))


if __name__ == "__main__":
    main()

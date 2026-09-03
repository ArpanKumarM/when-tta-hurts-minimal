"""Post-review extended analyses (see the paper's "Extended Analyses"
section), computed from the already-released final-test prediction assets
-- no retraining, no test-split re-opening.

Covers: the view-count scaling curve, the aggregation-rule ablation,
clean-anchoring, BatchNorm-statistics adaptation, and calibration
(ECE / NLL / Brier) -- all a pure re-aggregation of predictions.npz.

The label-preservation audit and the per-augmentation-component
decomposition are in the full repository (protocols under its
docs/phase2c*); the checkpoints and predictions bundled here are
sufficient to re-run them with that repo's scripts.

    uv run python analyze_extended.py
"""

import hashlib
import json

import numpy as np

import analyze
import config

PREFIXES = list(config.PREFIX_SEQUENCE)
OUT = config.RESULTS_ROOT / "extended_summary.json"
TABLE = config.RESULTS_ROOT / "tables" / "scaling_curve.csv"


def _seed(tag):
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


def _unmatched_headline_cells():
    # the 30 distinct unmatched-policy cells: Block A + Block D, policy none
    return [c for c in config.CELLS if c["block"] in ("A", "D") and c["training_policy"] == "none"]


def _metrics(run_id):
    path = config.PREDICTION_ROOT / run_id / "test" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def scaling_curve():
    rows = []
    for cell in _unmatched_headline_cells():
        preds = analyze.load_predictions(cell["run_id"])
        labels = preds["labels"]
        clean_correct = analyze.correctness(preds["clean_probs"], labels)
        for n in PREFIXES:
            tta_correct = analyze.correctness(analyze.mean_probability(preds["view_probs"], n), labels)
            boot = analyze.paired_bootstrap_ci(
                clean_correct, tta_correct, _seed(f"phase2c|scaling_curve|{cell['run_id']}|n{n}")
            )
            es = analyze.effect_sizes(clean_correct, tta_correct)
            rows.append({
                "run_id": cell["run_id"], "dataset": cell["dataset"],
                "resolution": cell["resolution"], "normalization": cell["normalization"],
                "seed": cell["seed"], "n_views": n,
                "delta_accuracy": boot["delta_accuracy"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "ci_excludes_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
                "harm_rate": es["harm_rate"], "rescue_rate": es["rescue_rate"],
            })
    return rows


def roll_up(rows):
    out = {}
    for n in PREFIXES:
        d = np.array([r["delta_accuracy"] for r in rows if r["n_views"] == n]) * 100
        out[str(n)] = {
            "n_cells": int(len(d)),
            "mean_delta_accuracy_pp": float(d.mean()),
            "min_delta_accuracy_pp": float(d.min()),
            "max_delta_accuracy_pp": float(d.max()),
            "n_cells_ci_excludes_zero_negative": int(
                sum(1 for r in rows if r["n_views"] == n and r["ci_excludes_zero"] and r["delta_accuracy"] < 0)
            ),
        }
    return out


def secondary_conditions():
    """Per-N deltas for the three source-study Appendix-B baselines and
    the aggregation-rule ablation, read from the per-cell metrics.json."""
    agg, anchored, bn, calib = {}, {}, {}, {}
    for cell in _unmatched_headline_cells():
        m = _metrics(cell["run_id"])
        if not m:
            continue
        rid = cell["run_id"]
        conds = m.get("conditions", {})
        for name in ("mean_probability", "majority_vote", "confidence_weighted_average"):
            block = (conds.get("naive_tta") or {}).get(name) or {}
            agg.setdefault(name, {})[rid] = {n: block[n]["delta_accuracy"] for n in block}
        if conds.get("original_anchored_tta"):
            anchored[rid] = {n: v["delta_accuracy"] for n, v in conds["original_anchored_tta"].items()}
        if conds.get("bn_adapted_tta"):
            bn[rid] = {n: v["delta_accuracy"] for n, v in conds["bn_adapted_tta"].items()}
        c = m.get("clean", {})
        n50 = (conds.get("naive_tta") or {}).get("mean_probability", {}).get("50", {})
        calib[rid] = {
            "clean": {k: c.get(k) for k in ("expected_calibration_error", "negative_log_likelihood", "brier_score")},
            "naive_tta_n50": {k: n50.get(k) for k in ("expected_calibration_error", "negative_log_likelihood", "brier_score")},
        }
    return {"aggregation_ablation": agg, "anchored": anchored, "bn_adapted": bn, "calibration": calib}


def main():
    rows = scaling_curve()
    (config.RESULTS_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with TABLE.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    summary = {
        "prefixes": PREFIXES,
        "scaling_curve_headline30_by_n": roll_up(rows),
        "scaling_curve_rows": rows,
        "secondary_conditions": secondary_conditions(),
        "note": "label-preservation audit and per-component decomposition are in the full repo",
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"wrote {OUT} and {TABLE}")
    cs = summary["scaling_curve_headline30_by_n"]
    for n in PREFIXES:
        d = cs[str(n)]
        print(f"  N={n:<3} mean {d['mean_delta_accuracy_pp']:+.2f}pp  "
              f"[{d['min_delta_accuracy_pp']:+.2f}, {d['max_delta_accuracy_pp']:+.2f}]  "
              f"CI<0 {d['n_cells_ci_excludes_zero_negative']}/{d['n_cells']}")


if __name__ == "__main__":
    main()

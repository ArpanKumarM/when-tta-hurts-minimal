import json
import math

import numpy as np

import config


def load_predictions(run_id, split="test"):
    path = config.PREDICTION_ROOT / run_id / split / "predictions.npz"
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def correctness(probs, labels):
    return probs.argmax(axis=-1) == labels


def mean_probability(view_probs, n):
    return view_probs[:n].mean(axis=0)


def paired_bootstrap_ci(clean_correct, tta_correct, seed, n_resamples=config.BOOTSTRAP_N_RESAMPLES, ci_level=config.BOOTSTRAP_CI_LEVEL):
    rng = np.random.default_rng(seed)
    n = clean_correct.size
    point = float(tta_correct.mean() - clean_correct.mean())
    resampled = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = tta_correct[idx].mean() - clean_correct[idx].mean()
    alpha = 1.0 - ci_level
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_accuracy": point, "ci_level": ci_level, "ci_low": float(lo), "ci_high": float(hi),
        "n_resamples": n_resamples, "n_samples": n, "bootstrap_seed": seed,
    }


def mcnemar_test(clean_correct, tta_correct):
    b = int(np.sum(clean_correct & ~tta_correct))
    c = int(np.sum(~clean_correct & tta_correct))
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "method": "undefined", "statistic": None, "p_value": None}
    if n_discordant < 25:
        k = min(b, c)
        p_value = min(1.0, 2.0 * sum(math.comb(n_discordant, x) * (0.5**n_discordant) for x in range(k + 1)))
        return {"b": b, "c": c, "n_discordant": n_discordant, "method": "exact_binomial", "statistic": None, "p_value": p_value}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {"b": b, "c": c, "n_discordant": n_discordant, "method": "continuity_corrected_chi_square", "statistic": float(statistic), "p_value": float(p_value)}


def benjamini_hochberg(p_values):
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        q = p_values[i] * n / k
        prev = min(prev, q)
        corrected[i] = prev
    return corrected


def effect_sizes(clean_correct, tta_correct):
    n_clean_correct = int(clean_correct.sum())
    n_clean_wrong = int((~clean_correct).sum())
    harmed = int(np.sum(clean_correct & ~tta_correct))
    rescued = int(np.sum(~clean_correct & tta_correct))
    return {
        "delta_accuracy": float(tta_correct.mean() - clean_correct.mean()),
        "harm_rate": (harmed / n_clean_correct) if n_clean_correct > 0 else 0.0,
        "rescue_rate": (rescued / n_clean_wrong) if n_clean_wrong > 0 else 0.0,
    }


def did_point_estimate(clean_a, tta_a, clean_b, tta_b):
    d = (tta_b.astype(np.float64) - clean_b.astype(np.float64)) - (tta_a.astype(np.float64) - clean_a.astype(np.float64))
    return float(d.mean())


def did_bootstrap_ci(clean_a, tta_a, clean_b, tta_b, seed, n_resamples=config.BOOTSTRAP_N_RESAMPLES, ci_level=config.BOOTSTRAP_CI_LEVEL):
    rng = np.random.default_rng(seed)
    n = clean_a.size
    point = did_point_estimate(clean_a, tta_a, clean_b, tta_b)
    resampled = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = did_point_estimate(clean_a[idx], tta_a[idx], clean_b[idx], tta_b[idx])
    alpha = 1.0 - ci_level
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"did": point, "ci_level": ci_level, "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples, "n_samples": n, "bootstrap_seed": seed}


FAMILY_BLOCKS = {
    "H1": lambda c: c["block"] == "A",
    "H2": lambda c: c["block"] in ("A", "D"),
    "BLOCK_C": lambda c: c["block"] == "C",
}


def family_cells(family):
    if family == "H3":
        matched = [c for c in config.CELLS if c["block"] == "B"]
        unmatched = []
        for m in matched:
            for c in config.CELLS:
                if c["block"] == "A" and c["dataset"] == m["dataset"] and c["resolution"] == m["resolution"] and c["normalization"] == m["normalization"] and c["seed"] == m["seed"]:
                    unmatched.append(c)
        return sorted(matched + unmatched, key=lambda c: c["run_id"])
    return sorted([c for c in config.CELLS if FAMILY_BLOCKS[family](c)], key=lambda c: c["run_id"])


def build_cell_result(family, cell):
    preds = load_predictions(cell["run_id"])
    labels = preds["labels"]
    clean_probs = preds["clean_probs"]
    view_probs = preds["view_probs"]
    clean_correct = correctness(clean_probs, labels)
    tta_correct = correctness(mean_probability(view_probs, config.PRIMARY_N), labels)
    seed = config.CELL_BOOTSTRAP_SEEDS[cell["run_id"]][family]
    return {
        "run_id": cell["run_id"],
        "family": family,
        "n_samples": int(len(labels)),
        "bootstrap": paired_bootstrap_ci(clean_correct, tta_correct, seed),
        "mcnemar": mcnemar_test(clean_correct, tta_correct),
        "effect_sizes": effect_sizes(clean_correct, tta_correct),
    }


def build_family(family):
    cells_result = [build_cell_result(family, c) for c in family_cells(family)]
    raw_p = [c["mcnemar"]["p_value"] if c["mcnemar"]["p_value"] is not None else 1.0 for c in cells_result]
    corrected_p = benjamini_hochberg(raw_p)
    return {
        "n_cells": len(cells_result),
        "cells": cells_result,
        "multiplicity": {"method": "benjamini_hochberg", "raw_p_values": raw_p, "corrected_p_values": corrected_p},
    }


def build_pair_result(pair):
    preds_a = load_predictions(pair["condition_a_run_id"])
    preds_b = load_predictions(pair["condition_b_run_id"])
    labels = preds_a["labels"]
    clean_a = correctness(preds_a["clean_probs"], labels)
    tta_a = correctness(mean_probability(preds_a["view_probs"], config.PRIMARY_N), labels)
    clean_b = correctness(preds_b["clean_probs"], labels)
    tta_b = correctness(mean_probability(preds_b["view_probs"], config.PRIMARY_N), labels)
    seed = config.PAIR_BOOTSTRAP_SEEDS[pair["pair_id"]]
    return {
        "pair_id": pair["pair_id"],
        "hypothesis": pair["hypothesis"],
        "condition_a": {"run_id": pair["condition_a_run_id"]},
        "condition_b": {"run_id": pair["condition_b_run_id"]},
        "n_samples": int(len(labels)),
        "bootstrap": did_bootstrap_ci(clean_a, tta_a, clean_b, tta_b, seed),
    }


def build_hypothesis_pairs(hypothesis):
    pairs = sorted([p for p in config.PAIRS if p["hypothesis"] == hypothesis], key=lambda p: p["pair_id"])
    pair_results = [build_pair_result(p) for p in pairs]
    return {"n_pairs": len(pair_results), "pairs": pair_results}


def build_summary():
    preregistered = {family: build_family(family) for family in ("H1", "H2", "H3", "BLOCK_C")}
    secondary = {hyp: build_hypothesis_pairs(hyp) for hyp in ("H1", "H2", "H3")}
    return {"schema_version": "minimal-v1", "preregistered": preregistered, "secondary_cross_condition": secondary}


def render_csv_rows(rows, header):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(h, "")) for h in header))
    return "\n".join(lines) + "\n"


def write_tables(summary, out_dir=config.RESULTS_ROOT / "tables"):
    out_dir.mkdir(parents=True, exist_ok=True)

    unmatched_by_run = {}
    for family in ("H1", "H2", "H3"):
        for cell in summary["preregistered"][family]["cells"]:
            if "matched_mixed" in cell["run_id"]:
                continue
            row = unmatched_by_run.setdefault(cell["run_id"], {"run_id": cell["run_id"], "families": []})
            row["families"].append(family)
            row[f"delta_accuracy_{family}"] = cell["bootstrap"]["delta_accuracy"]
    rows = sorted(unmatched_by_run.values(), key=lambda r: r["run_id"])
    (out_dir / "table_2_unmatched_policy.csv").write_text(
        render_csv_rows(rows, ["run_id", "families", "delta_accuracy_H1", "delta_accuracy_H2", "delta_accuracy_H3"])
    )

    block_c_rows = [
        {"run_id": c["run_id"], "delta_accuracy": c["bootstrap"]["delta_accuracy"], "ci_low": c["bootstrap"]["ci_low"], "ci_high": c["bootstrap"]["ci_high"]}
        for c in summary["preregistered"]["BLOCK_C"]["cells"]
    ]
    (out_dir / "table_6_block_c.csv").write_text(render_csv_rows(block_c_rows, ["run_id", "delta_accuracy", "ci_low", "ci_high"]))

    for hyp, name in (("H1", "table_4_normalization"), ("H2", "table_5_resolution"), ("H3", "table_3_matched_policy")):
        pair_rows = [
            {"pair_id": p["pair_id"], "did": p["bootstrap"]["did"], "ci_low": p["bootstrap"]["ci_low"], "ci_high": p["bootstrap"]["ci_high"]}
            for p in summary["secondary_cross_condition"][hyp]["pairs"]
        ]
        (out_dir / f"{name}.csv").write_text(render_csv_rows(pair_rows, ["pair_id", "did", "ci_low", "ci_high"]))


def main():
    summary = build_summary()
    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_tables(summary)
    print("wrote", config.RESULTS_ROOT / "summary.json")


if __name__ == "__main__":
    main()

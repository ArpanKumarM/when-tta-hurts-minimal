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


class _RowShapeError(ValueError):
    pass


def parse_run_id(run_id):
    parts = run_id.split("-")
    if "policy" not in parts or not run_id.split("-s")[-1].isdigit():
        raise _RowShapeError(run_id)
    dataset = parts[1]
    resolution = parts[2].removesuffix("px")
    policy_idx = parts.index("policy")
    normalization = "-".join(parts[3:policy_idx])
    policy = parts[policy_idx + 1]
    seed = run_id.rsplit("-s", 1)[-1]
    return {"dataset": dataset, "resolution": resolution, "normalization": normalization, "policy": policy, "seed": seed}


def cell_row(family, cell, index, multiplicity):
    identity = parse_run_id(cell["run_id"])
    return {
        "run_id": cell["run_id"],
        **identity,
        "delta_accuracy": cell["bootstrap"]["delta_accuracy"],
        "ci_low": cell["bootstrap"]["ci_low"],
        "ci_high": cell["bootstrap"]["ci_high"],
        "mcnemar_p": multiplicity["raw_p_values"][index],
        "bh_adjusted_p": multiplicity["corrected_p_values"][index],
    }


def extract_unmatched_cells(summary):
    by_run_id = {}
    for family in ("H1", "H2", "H3"):
        cells = summary["preregistered"][family]["cells"]
        multiplicity = summary["preregistered"][family]["multiplicity"]
        for index, cell in enumerate(cells):
            identity = parse_run_id(cell["run_id"])
            if identity["policy"] != "none":
                continue
            run_id = cell["run_id"]
            if run_id not in by_run_id:
                row = cell_row(family, cell, index, multiplicity)
                row["member_families"] = [family]
                row["bh_adjusted_p_by_family"] = {family: row.pop("bh_adjusted_p")}
                by_run_id[run_id] = row
            else:
                by_run_id[run_id]["member_families"].append(family)
                by_run_id[run_id]["bh_adjusted_p_by_family"][family] = multiplicity["corrected_p_values"][index]
    return sorted(by_run_id.values(), key=lambda r: (r["dataset"], int(r["resolution"]), r["normalization"], int(r["seed"])))


def extract_matched_within_cell(summary):
    cells = summary["preregistered"]["H3"]["cells"]
    multiplicity = summary["preregistered"]["H3"]["multiplicity"]
    rows = []
    for index, cell in enumerate(cells):
        if parse_run_id(cell["run_id"])["policy"] != "matched_mixed":
            continue
        rows.append(cell_row("H3", cell, index, multiplicity))
    return sorted(rows, key=lambda r: (r["dataset"], int(r["seed"])))


def extract_cross_condition_pairs(summary, hypothesis):
    return sorted(summary["secondary_cross_condition"][hypothesis]["pairs"], key=lambda p: p["pair_id"])


def extract_block_c(summary):
    cells = summary["preregistered"]["BLOCK_C"]["cells"]
    multiplicity = summary["preregistered"]["BLOCK_C"]["multiplicity"]
    rows = [cell_row("BLOCK_C", c, i, multiplicity) for i, c in enumerate(cells)]
    return sorted(rows, key=lambda r: int(r["seed"]))


def pp(value):
    return f"{value * 100:.2f}"


def render_design_classification_table():
    lines = [
        "# Table 1 — Experimental-Design and Evidence-Classification",
        "",
        "| Evidence tier | Source | Cells/pairs | Confirmatory? |",
        "|---|---|---|---|",
        "| Preregistered within-cell | H1/H2/H3/BLOCK_C (`preregistered.*`) "
        "| 39 unique cells | Yes -- clean-vs-TTA, within one trained model |",
        "| Secondary fixed-model comparison | Cross-condition H1/H2/H3 (`secondary_cross_condition.*`) "
        "| 30 pairs (12+12+6) | No -- post-validation/pre-test-specified, never preregistered |",
        "| Descriptive summary | `descriptive_summaries.preregistered_seed_level` "
        "| 13 dataset/resolution/normalization groups "
        "| No -- non-inferential, carries no p-value/CI of its own |",
        "| Unsupported/forbidden | H4; pooled/model-population verdicts; secondary significance labels "
        "| N/A | Never permitted anywhere in this package |",
        "",
    ]
    return "\n".join(lines)


def render_unmatched_table(rows):
    lines = [
        "# Table 2 — Complete 30-Cell Unmatched-Policy Table",
        "",
        "Preregistered within-cell evidence. Every distinct unmatched-policy "
        "cell appears exactly once, regardless of how many hypothesis "
        "families (listed in `member_families`) it belongs to.",
        "",
        "| run_id | dataset | resolution | normalization | seed | Δ accuracy (pp) | 95% CI (pp) "
        "| McNemar p | member families | BH-adjusted p (per family) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"[{pp(r['ci_low'])}, {pp(r['ci_high'])}]"
        bh = "; ".join(f"{fam}={p:.3g}" for fam, p in sorted(r["bh_adjusted_p_by_family"].items()))
        lines.append(
            f"| {r['run_id']} | {r['dataset']} | {r['resolution']}px | {r['normalization']} | "
            f"{r['seed']} | {pp(r['delta_accuracy'])} | {ci} | {r['mcnemar_p']:.3g} | "
            f"{', '.join(r['member_families'])} | {bh} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_matched_table(within_cell, pairs):
    lines = [
        "# Table 3 — Matched-Policy Within-Cell and Secondary DiD Table",
        "",
        "## Within-cell (preregistered, H3 matched arm)",
        "",
        "| run_id | dataset | seed | Δ accuracy (pp) | 95% CI (pp) | McNemar p | BH-adjusted p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in within_cell:
        ci = f"[{pp(r['ci_low'])}, {pp(r['ci_high'])}]"
        lines.append(
            f"| {r['run_id']} | {r['dataset']} | {r['seed']} | {pp(r['delta_accuracy'])} | {ci} | "
            f"{r['mcnemar_p']:.3g} | {r['bh_adjusted_p']:.3g} |"
        )
    lines.append("")
    lines.append(
        "## Secondary (post-validation/pre-test-specified, fixed-model DiD -- not a preregistered "
        "cross-condition test)"
    )
    lines.append("")
    lines.append("| pair_id | condition A | condition B | DiD (pp) | 95% CI (pp) |")
    lines.append("|---|---|---|---|---|")
    for p in pairs:
        ci = f"[{pp(p['bootstrap']['ci_low'])}, {pp(p['bootstrap']['ci_high'])}]"
        lines.append(
            f"| {p['pair_id']} | {p['condition_a']['run_id']} | {p['condition_b']['run_id']} | "
            f"{pp(p['bootstrap']['did'])} | {ci} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_cross_condition_table(title, pairs):
    lines = [
        f"# {title}",
        "",
        "Secondary, fixed-model, post-validation/pre-test-specified "
        "difference-in-differences estimates. No pooled p-value, alpha "
        "threshold, or significance decision is computed or implied.",
        "",
        "| pair_id | condition A | condition B | DiD (pp) | 95% CI (pp) |",
        "|---|---|---|---|---|",
    ]
    for p in pairs:
        ci = f"[{pp(p['bootstrap']['ci_low'])}, {pp(p['bootstrap']['ci_high'])}]"
        lines.append(
            f"| {p['pair_id']} | {p['condition_a']['run_id']} | {p['condition_b']['run_id']} | "
            f"{pp(p['bootstrap']['did'])} | {ci} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_block_c_table(rows):
    lines = [
        "# Table 6 — Complete Three-Seed BLOCK_C Table",
        "",
        "| run_id | seed | Δ accuracy (pp) | 95% CI (pp) | McNemar p | BH-adjusted p |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"[{pp(r['ci_low'])}, {pp(r['ci_high'])}]"
        lines.append(
            f"| {r['run_id']} | {r['seed']} | {pp(r['delta_accuracy'])} | {ci} | "
            f"{r['mcnemar_p']:.3g} | {r['bh_adjusted_p']:.3g} |"
        )
    lines.append("")
    lines.append(
        "External reference (descriptive only, not an acceptance threshold): the source paper's own "
        "reported TTA improvement at N=50 views was approximately +1.6 percentage points. This "
        "project's frozen operationalization did not reproduce that expected positive improvement in "
        "any of the three seeds above."
    )
    lines.append("")
    return "\n".join(lines)


def render_claim_adjudication_table():
    lines = [
        "# Table 7 — Claim Adjudication",
        "",
        "| Claim | Evidence tier | Status |",
        "|---|---|---|",
        "| Naive TTA harmed all 30 distinct unmatched-policy base cells "
        "| Preregistered within-cell | Supported |",
        "| Matched-policy training mitigates TTA harm "
        "| Secondary fixed-model DiD, descriptively corroborated by separate within-cell patterns "
        "| Supported only secondarily/descriptively -- not a preregistered cross-condition test |",
        "| Normalization changes the magnitude of harm | Secondary fixed-model DiD only "
        "| Supported only secondarily; direction is dataset-dependent |",
        "| Higher resolution reduces TTA harm | Secondary fixed-model DiD only "
        "| Contradicted for BloodMNIST; mixed/near-null for PathMNIST |",
        "| BLOCK_C reproduces the source paper's positive TTA improvement "
        "| Preregistered within-cell (positive control) "
        "| Contradicted -- expected positive improvement not reproduced in any seed |",
        "| Any H4 (Validation-Gated TTA) verdict | None -- no derivable family exists | Not made, anywhere |",
        "| Any model-population or general medical-imaging generalization "
        "| None -- three seeds, fixed policy/budget | Not permitted, anywhere |",
        "",
    ]
    return "\n".join(lines)


def markdown_table_to_csv(markdown_text):
    import csv
    import io
    import re

    table_lines = [line for line in markdown_text.splitlines() if line.strip().startswith("|")]
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    rows = [r for r in rows if not all(re.fullmatch(r"-+", c) for c in r)]
    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def write_tables(summary, out_dir=None):
    out_dir = out_dir or (config.RESULTS_ROOT / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    unmatched = extract_unmatched_cells(summary)
    matched = extract_matched_within_cell(summary)
    h1_pairs = extract_cross_condition_pairs(summary, "H1")
    h2_pairs = extract_cross_condition_pairs(summary, "H2")
    h3_pairs = extract_cross_condition_pairs(summary, "H3")
    block_c = extract_block_c(summary)

    tables = {
        "table_1_design_classification": render_design_classification_table(),
        "table_2_unmatched_policy": render_unmatched_table(unmatched),
        "table_3_matched_policy": render_matched_table(matched, h3_pairs),
        "table_4_normalization": render_cross_condition_table("Table 4 — Complete 12-Pair Normalization Table", h1_pairs),
        "table_5_resolution": render_cross_condition_table("Table 5 — Complete 12-Pair Resolution Table", h2_pairs),
        "table_6_block_c": render_block_c_table(block_c),
        "table_7_claim_adjudication": render_claim_adjudication_table(),
    }
    for name, markdown_text in tables.items():
        (out_dir / f"{name}.csv").write_text(markdown_table_to_csv(markdown_text))

    return unmatched, matched, h1_pairs, h2_pairs, h3_pairs, block_c


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

_DETERMINISTIC_PDF_METADATA = {"CreationDate": None, "Creator": "", "Producer": "", "Author": ""}
_DETERMINISTIC_PNG_METADATA = {"Software": ""}


def _save_deterministic(fig, path_no_ext):
    pdf_path = path_no_ext.with_suffix(".pdf")
    png_path = path_no_ext.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", metadata=_DETERMINISTIC_PDF_METADATA)
    fig.savefig(png_path, format="png", dpi=150, metadata=_DETERMINISTIC_PNG_METADATA)
    return pdf_path, png_path


def _forest_plot(fig_ax, rows, labels, values_key, ci_low_key, ci_high_key, row_colors, xlabel, caption, title):
    fig, ax = fig_ax
    n = len(rows)
    y_positions = list(range(n))
    values = [r[values_key] * 100 for r in rows]
    ci_low = [r[ci_low_key] * 100 for r in rows]
    ci_high = [r[ci_high_key] * 100 for r in rows]
    err_low = [v - lo for v, lo in zip(values, ci_low)]
    err_high = [hi - v for v, hi in zip(values, ci_high)]
    for i, color in enumerate(row_colors):
        ax.errorbar(
            [values[i]], [y_positions[i]], xerr=[[err_low[i]], [err_high[i]]], fmt="o", markersize=4,
            capsize=3, ecolor=color, markerfacecolor=color, markeredgecolor=OKABE_ITO["black"], linewidth=1.2,
        )
    ax.axvline(0.0, color=OKABE_ITO["black"], linewidth=1.0, linestyle="-")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.tick_params(labelsize=9)
    if caption:
        fig.text(0.02, 0.01, caption, fontsize=7, wrap=True, ha="left", va="bottom")


def render_figure_1(rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_by_norm = {"batchnorm": OKABE_ITO["blue"], "groupnorm": OKABE_ITO["vermillion"]}
    labels = [f"{r['dataset']} {r['resolution']}px {r['normalization']} s{r['seed']}" for r in rows]
    colors = [color_by_norm.get(r["normalization"], OKABE_ITO["black"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(rows) + 1.0)))
    _forest_plot(
        (fig, ax), rows, labels, "delta_accuracy", "ci_low", "ci_high", colors,
        "Δ accuracy, TTA − clean (pp)",
        "Preregistered within-cell clean-versus-TTA evidence. Each row is one trained model "
        "(dataset x resolution x normalization x seed); no cross-condition comparison is made or "
        "implied here.",
        "Figure 1 — Unmatched-policy TTA effects (30 cells, preregistered)",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_1_unmatched_policy_forest")
    plt.close(fig)
    return paths


def render_figure_2(pairs, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [{"did": p["bootstrap"]["did"], "ci_low": p["bootstrap"]["ci_low"], "ci_high": p["bootstrap"]["ci_high"]} for p in pairs]
    labels = [p["pair_id"] for p in pairs]
    colors = [OKABE_ITO["bluish_green"]] * len(pairs)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.35 * len(pairs) + 1.0)))
    _forest_plot(
        (fig, ax), rows, labels, "did", "ci_low", "ci_high", colors,
        "DiD, matched − unmatched policy (pp)",
        "Secondary, fixed-model, post-validation/pre-test-specified difference-in-differences "
        "comparison -- not a preregistered cross-condition inference. No significance decision is "
        "made for these estimates.",
        "Figure 2 — Matched-policy mitigation (6 secondary DiD pairs)",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_2_matched_policy_mitigation")
    plt.close(fig)
    return paths


def render_figure_3(pairs, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def dataset_of(p):
        return parse_run_id(p["condition_a"]["run_id"])["dataset"]

    datasets = sorted({dataset_of(p) for p in pairs})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 5.0), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        subset = sorted((p for p in pairs if dataset_of(p) == dataset), key=lambda p: p["pair_id"])
        rows = [{"did": p["bootstrap"]["did"], "ci_low": p["bootstrap"]["ci_low"], "ci_high": p["bootstrap"]["ci_high"]} for p in subset]
        labels = [p["pair_id"] for p in subset]
        colors = [OKABE_ITO["sky_blue"]] * len(subset)
        _forest_plot((fig, ax), rows, labels, "did", "ci_low", "ci_high", colors, "DiD, GroupNorm − BatchNorm (pp)", "", dataset)
    fig.suptitle("Figure 3 — Normalization heterogeneity (12 secondary DiD pairs)", fontsize=10)
    fig.text(
        0.02, 0.01,
        "The direction of this secondary estimate is dataset-dependent (see panels) and must not be "
        "read as a general BatchNorm-vs-GroupNorm verdict.",
        fontsize=7, wrap=True, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    paths = _save_deterministic(fig, output_dir / "figure_3_normalization_heterogeneity")
    plt.close(fig)
    return paths


def render_figure_4(pairs, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def group_of(p):
        identity = parse_run_id(p["condition_a"]["run_id"])
        return f"{identity['dataset']}/{identity['normalization']}"

    sorted_pairs = sorted(pairs, key=lambda p: (group_of(p), p["pair_id"]))
    labels = [f"{group_of(p)} {p['pair_id']}" for p in sorted_pairs]
    rows = [{"did": p["bootstrap"]["did"], "ci_low": p["bootstrap"]["ci_low"], "ci_high": p["bootstrap"]["ci_high"]} for p in sorted_pairs]
    colors = [OKABE_ITO["orange"]] * len(sorted_pairs)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(sorted_pairs) + 1.0)))
    _forest_plot(
        (fig, ax), rows, labels, "did", "ci_low", "ci_high", colors,
        "DiD, high-res − low-res (pp)",
        "BloodMNIST pairs trend contrary to the hypothesized direction; PathMNIST pairs are "
        "mixed/near-null. Neither pattern is a preregistered or confirmatory test of H2. The dashed "
        "vertical line marks the hypothesized positive direction (reference only, not a confirmation "
        "marker).",
        "Figure 4 — Resolution comparison (12 secondary DiD pairs)",
    )
    ax.axvline(0.0, color=OKABE_ITO["black"], linewidth=1.0)
    ax.axvline(1.0, color=OKABE_ITO["reddish_purple"], linewidth=1.0, linestyle="--")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_4_resolution_comparison")
    plt.close(fig)
    return paths


def render_figure_5(rows, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"seed {r['seed']}" for r in rows]
    colors = [OKABE_ITO["blue"]] * len(rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    _forest_plot(
        (fig, ax), rows, labels, "delta_accuracy", "ci_low", "ci_high", colors,
        "Δ accuracy, TTA − clean (pp)",
        "The expected positive TTA improvement (~+1.6pp) was not reproduced in this project's "
        "frozen operationalization. The dashed vertical line is the external reference (source "
        "paper), descriptive only -- not an acceptance threshold.",
        "Figure 5 — BLOCK_C positive control (3 seeds, preregistered)",
    )
    ax.axvline(1.6, color=OKABE_ITO["reddish_purple"], linewidth=1.0, linestyle="--")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_5_block_c_positive_control")
    plt.close(fig)
    return paths


def write_figures(unmatched, matched, h1_pairs, h2_pairs, h3_pairs, block_c, out_dir=None):
    out_dir = out_dir or (config.RESULTS_ROOT / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_figure_1(unmatched, out_dir)
    render_figure_2(h3_pairs, out_dir)
    render_figure_3(h1_pairs, out_dir)
    render_figure_4(h2_pairs, out_dir)
    render_figure_5(block_c, out_dir)


def main():
    summary = build_summary()
    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    unmatched, matched, h1_pairs, h2_pairs, h3_pairs, block_c = write_tables(summary)
    write_figures(unmatched, matched, h1_pairs, h2_pairs, h3_pairs, block_c)
    print("wrote", config.RESULTS_ROOT)


if __name__ == "__main__":
    main()

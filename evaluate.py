import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import config
import data
import model as model_lib

_EPS = 1e-12


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def accuracy(probs, labels):
    return float((probs.argmax(axis=-1) == labels).mean())


def macro_f1(probs, labels):
    preds = probs.argmax(axis=-1)
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def nll(probs, labels, eps=_EPS):
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))


def ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        total += (count / n) * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(total)


def brier(probs, labels):
    n, c = probs.shape
    one_hot = np.zeros((n, c))
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def harm_rescue_rates(clean_probs, tta_probs, labels):
    clean_correct = clean_probs.argmax(axis=-1) == labels
    tta_correct = tta_probs.argmax(axis=-1) == labels
    n_clean_correct = int(clean_correct.sum())
    n_clean_wrong = int((~clean_correct).sum())
    harmed = clean_correct & ~tta_correct
    rescued = ~clean_correct & tta_correct
    harm_rate = float(harmed.sum() / n_clean_correct) if n_clean_correct > 0 else 0.0
    rescue_rate = float(rescued.sum() / n_clean_wrong) if n_clean_wrong > 0 else 0.0
    return harm_rate, rescue_rate


def per_prefix_metrics(clean_probs, agg_probs, labels):
    harm_rate, rescue_rate = harm_rescue_rates(clean_probs, agg_probs, labels)
    return {
        "accuracy": accuracy(agg_probs, labels),
        "macro_f1": macro_f1(agg_probs, labels),
        "negative_log_likelihood": nll(agg_probs, labels),
        "expected_calibration_error": ece(agg_probs, labels),
        "brier_score": brier(agg_probs, labels),
        "delta_accuracy": accuracy(agg_probs, labels) - accuracy(clean_probs, labels),
        "harm_rate": harm_rate,
        "rescue_rate": rescue_rate,
    }


def mean_probability(view_probs, n_views):
    return view_probs[:n_views].mean(axis=0)


def majority_vote(view_probs, n_views):
    prefix = view_probs[:n_views]
    n_samples, n_classes = prefix.shape[1], prefix.shape[2]
    view_preds = prefix.argmax(axis=-1)
    votes = np.zeros((n_samples, n_classes), dtype=np.int64)
    for v in range(n_views):
        for c in range(n_classes):
            votes[:, c] += view_preds[v] == c
    mean_probs = prefix.mean(axis=0)
    predicted = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        max_votes = votes[i].max()
        tied = np.flatnonzero(votes[i] == max_votes)
        if len(tied) == 1:
            predicted[i] = tied[0]
        else:
            best = mean_probs[i, tied].max()
            still_tied = tied[mean_probs[i, tied] == best]
            predicted[i] = still_tied.min()
    return votes.astype(np.float64) / n_views


def confidence_weighted_average(view_probs, n_views):
    prefix = view_probs[:n_views]
    confidences = prefix.max(axis=-1)
    weights = confidences / confidences.sum(axis=0, keepdims=True)
    return (prefix * weights[:, :, None]).sum(axis=0)


def original_anchored_mean_probability(clean_probs, view_probs, n_views):
    aug = view_probs[:n_views]
    all_probs = np.concatenate([clean_probs[None, :, :], aug], axis=0)
    return all_probs.mean(axis=0)


def bn_adapt_sequential(net, microbatches):
    if not model_lib.has_batchnorm(net):
        return None
    adapted = copy.deepcopy(net)
    adapted.train()
    with torch.no_grad():
        for batch in microbatches:
            adapted(batch)
    adapted.eval()
    return adapted


def bn_adaptation_microbatches(images_cpu, policy, tta_seed, dataset, resolution, sample_indices, n, device, batch_size):
    for _view_index, view_batch in data.iter_deterministic_views(
        images_cpu, policy, tta_seed, dataset, resolution, sample_indices, n
    ):
        view_n = view_batch.shape[0]
        for start in range(0, view_n, batch_size):
            yield view_batch[start : start + batch_size].to(device)


def load_checkpoint(cell, device):
    n_classes = config.DATASETS[cell["dataset"]]["n_classes"]
    net = model_lib.build_model(cell, n_classes)
    ckpt_path = config.CHECKPOINT_ROOT / cell["run_id"] / f"attempt_{cell['attempt']:03d}" / "best_checkpoint.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net.to(device)


def evaluate_cell(cell, split, device):
    images, labels, sample_indices = data.load_split(cell["dataset"], cell["resolution"], split)
    net = load_checkpoint(cell, device)

    clean_logits = data.batched_forward(net, images, device, config.INFERENCE_BATCH_SIZE)
    clean_probs = softmax(clean_logits)
    n_classes = clean_probs.shape[-1]

    policy = data.build_policy(config.POLICY_IDENTIFIER, (cell["resolution"], cell["resolution"]))

    view_probs = np.empty((config.MAX_VIEWS, len(sample_indices), n_classes), dtype=np.float32)
    for view_index, view_batch in data.iter_deterministic_views(
        images, policy, config.TTA_SEED, cell["dataset"], cell["resolution"], sample_indices, config.MAX_VIEWS
    ):
        view_logits = data.batched_forward(net, view_batch, device, config.INFERENCE_BATCH_SIZE)
        view_probs[view_index] = softmax(view_logits)
        del view_batch

    conditions = {"naive_tta": {agg: {} for agg in config.AGGREGATORS}}
    for n in config.PREFIX_SEQUENCE:
        conditions["naive_tta"]["mean_probability"][n] = per_prefix_metrics(
            clean_probs, mean_probability(view_probs, n), labels
        )
        conditions["naive_tta"]["majority_vote"][n] = per_prefix_metrics(
            clean_probs, majority_vote(view_probs, n), labels
        )
        conditions["naive_tta"]["confidence_weighted_average"][n] = per_prefix_metrics(
            clean_probs, confidence_weighted_average(view_probs, n), labels
        )

    conditions["original_anchored_tta"] = {
        n: per_prefix_metrics(clean_probs, original_anchored_mean_probability(clean_probs, view_probs, n), labels)
        for n in config.PREFIX_SEQUENCE
    }

    bn_probs_by_n = {}
    if model_lib.has_batchnorm(net):
        conditions["bn_adapted_tta"] = {}
        for n in config.PREFIX_SEQUENCE:
            microbatches = bn_adaptation_microbatches(
                images, policy, config.TTA_SEED, cell["dataset"], cell["resolution"], sample_indices, n,
                device, config.BN_ADAPTATION_BATCH_SIZE,
            )
            adapted = bn_adapt_sequential(net, microbatches)
            adapted_probs = softmax(data.batched_forward(adapted, images, device, config.INFERENCE_BATCH_SIZE))
            bn_probs_by_n[n] = adapted_probs
            conditions["bn_adapted_tta"][n] = per_prefix_metrics(clean_probs, adapted_probs, labels)
    else:
        conditions["bn_adapted_tta"] = None

    predictions = {
        "labels": labels,
        "sample_indices": sample_indices,
        "clean_probs": clean_probs.astype(np.float32),
        "view_probs": view_probs,
    }
    if bn_probs_by_n:
        predictions["bn_adapted_probs"] = np.stack(
            [bn_probs_by_n[n] for n in config.PREFIX_SEQUENCE], axis=0
        ).astype(np.float32)
        predictions["bn_adapted_prefix_sequence"] = np.array(config.PREFIX_SEQUENCE, dtype=np.int64)

    metrics = {
        "clean": {
            "accuracy": accuracy(clean_probs, labels),
            "macro_f1": macro_f1(clean_probs, labels),
            "negative_log_likelihood": nll(clean_probs, labels),
            "expected_calibration_error": ece(clean_probs, labels),
            "brier_score": brier(clean_probs, labels),
        },
        "conditions": conditions,
    }
    return predictions, metrics


def run_one(run_id, split, device):
    cell = next(c for c in config.CELLS if c["run_id"] == run_id)
    predictions, metrics = evaluate_cell(cell, split, device)
    out_dir = config.PREDICTION_ROOT / run_id / split
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "predictions.npz", **predictions)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(run_id, split, "accuracy(clean)=", metrics["clean"]["accuracy"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = "val" if args.split == "validation" else "test"

    if args.all:
        for cell in config.CELLS:
            run_one(cell["run_id"], split, device)
    elif args.run_id:
        run_one(args.run_id, split, device)
    else:
        parser.error("pass --run-id or --all")


if __name__ == "__main__":
    main()

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from torch import nn

import config
import data
import model as model_lib


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def evaluate_loss_accuracy(net, loader, device, criterion):
    net.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).long().view(-1)
            logits = net(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def train_cell(cell, device):
    seed_everything(cell["seed"])
    n_classes = config.DATASETS[cell["dataset"]]["n_classes"]
    net = model_lib.build_model(cell, n_classes).to(device)

    train_images, train_labels, _ = data.load_split(cell["dataset"], cell["resolution"], "train")
    val_images, val_labels, _ = data.load_split(cell["dataset"], cell["resolution"], "val")

    generator = torch.Generator()
    generator.manual_seed(cell["seed"])
    train_loader = data.make_loader(train_images, train_labels, config.TRAINING["batch_size"], True, generator)
    val_loader = data.make_loader(val_images, val_labels, config.TRAINING["batch_size"], False)

    augmentation_policy = None
    augmentation_seed = None
    if cell["training_policy"] == "matched_to_approved_tta_policy":
        augmentation_policy = data.build_policy(config.POLICY_IDENTIFIER, (cell["resolution"], cell["resolution"]))
        augmentation_seed = config.TTA_SEED

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.TRAINING["learning_rate"], weight_decay=config.TRAINING["weight_decay"]
    )
    max_epochs = config.TRAINING["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_accuracy = -1.0
    best_state_dict = None
    epochs_without_improvement = 0
    step_counter = 0
    patience = config.TRAINING["early_stopping_patience"]
    min_delta = config.TRAINING["early_stopping_min_delta"]

    for epoch in range(1, max_epochs + 1):
        net.train()
        for x, y in train_loader:
            if augmentation_policy is not None:
                x = data.sample_deterministic_view(x, augmentation_policy, seed=augmentation_seed + step_counter)
                step_counter += 1
            x, y = x.to(device), y.to(device).long().view(-1)
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        _, val_accuracy = evaluate_loss_accuracy(net, val_loader, device, criterion)
        if val_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = val_accuracy
            best_state_dict = copy.deepcopy(net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    net.load_state_dict(best_state_dict)
    return net, best_val_accuracy


def train_one(run_id, device):
    cell = next(c for c in config.CELLS if c["run_id"] == run_id)
    t0 = time.perf_counter()
    net, best_val_accuracy = train_cell(cell, device)
    elapsed = time.perf_counter() - t0
    out_dir = config.CHECKPOINT_ROOT / run_id / f"attempt_{cell['attempt']:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / "best_checkpoint.pt")
    print(f"{run_id} best_val_accuracy={best_val_accuracy:.4f} elapsed={elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if args.all:
        for cell in config.CELLS:
            train_one(cell["run_id"], device)
    elif args.run_id:
        train_one(args.run_id, device)
    else:
        parser.error("pass --run-id or --all")


if __name__ == "__main__":
    main()

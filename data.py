import hashlib
import urllib.request
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config

_FLIP_P = 0.5
_ROTATION_DEGREES = 15.0
_ROTATION_RESAMPLE = "BILINEAR"
_CROP_SCALE = (0.8, 1.0)
_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)
_CROP_RESAMPLE = "BILINEAR"
_JITTER_BRIGHTNESS = 0.3
_JITTER_CONTRAST = 0.3
_GAUSSIAN_BLUR_KERNEL = (3, 3)
_GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)
_GAUSSIAN_BLUR_P = 0.5
_SEED_MODULUS = 2**31 - 1


def _md5_of_file(path):
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def artifact_filename(dataset, resolution):
    return f"{dataset}.npz" if resolution == 28 else f"{dataset}_{resolution}.npz"


def ensure_dataset(dataset, resolution, root=config.DATA_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / artifact_filename(dataset, resolution)
    expected_md5 = config.DATASETS[dataset]["md5"][resolution]
    if path.exists() and _md5_of_file(path) == expected_md5:
        return path
    url = config.DATASETS[dataset]["urls"][resolution]
    urllib.request.urlretrieve(url, path)
    actual_md5 = _md5_of_file(path)
    if actual_md5 != expected_md5:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {path}: expected {expected_md5}, got {actual_md5}")
    return path


def load_split(dataset, resolution, split, root=config.DATA_ROOT):
    path = ensure_dataset(dataset, resolution, root)
    with np.load(path) as npz:
        images = npz[f"{split}_images"]
        labels = npz[f"{split}_labels"]
    images = torch.from_numpy(images).float() / 255.0
    if images.ndim == 3:
        images = images.unsqueeze(1)
    else:
        images = images.permute(0, 3, 1, 2)
    labels = labels.reshape(-1).astype(np.int64)
    sample_indices = np.arange(len(labels), dtype=np.int64)
    return images, labels, sample_indices


def geometric_ops(output_size):
    return [
        K.RandomHorizontalFlip(p=_FLIP_P, same_on_batch=False),
        K.RandomVerticalFlip(p=_FLIP_P, same_on_batch=False),
        K.RandomRotation(degrees=_ROTATION_DEGREES, resample=_ROTATION_RESAMPLE, p=1.0, same_on_batch=False),
        K.RandomResizedCrop(
            size=output_size, scale=_CROP_SCALE, ratio=_CROP_RATIO, resample=_CROP_RESAMPLE,
            p=1.0, same_on_batch=False,
        ),
    ]


def intensity_ops():
    return [
        K.ColorJitter(brightness=_JITTER_BRIGHTNESS, contrast=_JITTER_CONTRAST, p=1.0, same_on_batch=False),
        K.RandomGaussianBlur(
            kernel_size=_GAUSSIAN_BLUR_KERNEL, sigma=_GAUSSIAN_BLUR_SIGMA, p=_GAUSSIAN_BLUR_P,
            same_on_batch=False,
        ),
    ]


def build_policy(name, output_size):
    geometric = geometric_ops(output_size)
    intensity = intensity_ops()
    if name == "geometric":
        ops = geometric
    elif name == "intensity":
        ops = intensity
    elif name == "mixed":
        ops = geometric + intensity
    else:
        raise ValueError(name)
    return nn.Sequential(*ops)


def sample_deterministic_view(x, policy, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return policy(x)


def stable_view_seed(tta_seed, dataset, resolution, sample_index, view_index):
    payload = f"{tta_seed}|{dataset}|{resolution}|{sample_index}|{view_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def generate_single_view(x, policy, tta_seed, dataset, resolution, sample_indices, view_index):
    x_cpu = x.detach().to("cpu")
    policy_cpu = policy.to("cpu")
    n = x_cpu.shape[0]
    out = torch.empty_like(x_cpu)
    for i in range(n):
        seed = stable_view_seed(tta_seed, dataset, resolution, sample_indices[i], view_index)
        torch.manual_seed(seed)
        with torch.no_grad():
            transformed = policy_cpu(x_cpu[i : i + 1])
        out[i] = transformed[0]
    return out


def iter_deterministic_views(x, policy, tta_seed, dataset, resolution, sample_indices, n_views):
    for view_index in range(n_views):
        yield view_index, generate_single_view(x, policy, tta_seed, dataset, resolution, sample_indices, view_index)


def batched_forward(model, images_cpu, device, batch_size):
    n = images_cpu.shape[0]
    chunks = []
    for start in range(0, n, batch_size):
        batch = images_cpu[start : start + batch_size].to(device)
        with torch.no_grad():
            chunks.append(model(batch).detach().cpu().numpy())
        del batch
    return np.concatenate(chunks, axis=0)


def make_loader(images, labels, batch_size, shuffle, generator=None):
    dataset = torch.utils.data.TensorDataset(images, torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

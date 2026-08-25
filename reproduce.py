import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import config
import analyze

MANIFEST_PATH = Path("manifest.json")


def sha256_of_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_extract(tar, destination):
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise RuntimeError(f"unsafe archive entry rejected: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"unsafe archive entry rejected (link): {member.name}")
    tar.extractall(destination)


def _gh_authenticated():
    result = subprocess.run(["gh", "auth", "status"], capture_output=True)
    return result.returncode == 0


def _gh_release_download(repo, tag, filename, dest_dir):
    subprocess.run(
        ["gh", "release", "download", tag, "--repo", repo, "--pattern", filename, "--dir", str(dest_dir), "--clobber"],
        check=True,
    )


def ensure_assets():
    checkpoints_present = config.CHECKPOINT_ROOT.exists() and any(config.CHECKPOINT_ROOT.iterdir())
    predictions_present = config.PREDICTION_ROOT.exists() and any(config.PREDICTION_ROOT.iterdir())
    if checkpoints_present and predictions_present:
        return

    if not MANIFEST_PATH.exists():
        raise RuntimeError("assets/ missing and manifest.json not found; cannot download release archives.")
    manifest = json.loads(MANIFEST_PATH.read_text())

    if not _gh_authenticated():
        print("assets/ missing and no local checkpoints/predictions found.")
        print("Downloading requires an authenticated GitHub CLI. Run: gh auth login")
        raise SystemExit(1)

    download_dir = Path("assets/_downloads")
    download_dir.mkdir(parents=True, exist_ok=True)
    repo = manifest["release_repo"]
    tag = manifest["release_tag"]
    for entry in manifest["archives"]:
        dest = download_dir / entry["filename"]
        if not dest.exists() or sha256_of_file(dest) != entry["sha256"]:
            print(f"downloading {entry['filename']} from {repo}@{tag} ...")
            _gh_release_download(repo, tag, entry["filename"], download_dir)
        actual = sha256_of_file(dest)
        if actual != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for {entry['filename']}: expected {entry['sha256']}, got {actual}")
        with tarfile.open(dest, "r:gz") as tar:
            _safe_extract(tar, Path(entry["extract_to"]))


def regenerate_outputs():
    if config.RESULTS_ROOT.exists():
        shutil.rmtree(config.RESULTS_ROOT)
    analyze.main()


def verify_output_hashes(manifest):
    mismatches = []
    for rel_path, expected in manifest["canonical_outputs"].items():
        path = Path(rel_path)
        if not path.exists():
            mismatches.append(f"{rel_path}: missing")
            continue
        actual = sha256_of_file(path)
        if actual != expected:
            mismatches.append(f"{rel_path}: expected={expected} actual={actual}")
    return mismatches


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    ensure_assets()
    regenerate_outputs()
    mismatches = verify_output_hashes(manifest)

    if mismatches:
        print("FAIL:", len(mismatches), "output hash mismatch(es)")
        for m in mismatches:
            print(" -", m)
        return 1

    print("PASS: regenerated summary.json, all 7 tables, and all 10 figures match canonical hashes exactly.")
    print(f"  outputs verified: {len(manifest['canonical_outputs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

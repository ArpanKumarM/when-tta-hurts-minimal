import hashlib
import json
import sys
import tarfile
import urllib.request
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


def ensure_assets():
    checkpoints_present = config.CHECKPOINT_ROOT.exists() and any(config.CHECKPOINT_ROOT.iterdir())
    predictions_present = config.PREDICTION_ROOT.exists() and any(config.PREDICTION_ROOT.iterdir())
    if checkpoints_present and predictions_present:
        return

    if not MANIFEST_PATH.exists():
        raise RuntimeError("assets/ missing and manifest.json not found; cannot download release archives.")
    manifest = json.loads(MANIFEST_PATH.read_text())

    download_dir = Path("assets/_downloads")
    download_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["archives"]:
        dest = download_dir / entry["filename"]
        if not dest.exists() or sha256_of_file(dest) != entry["sha256"]:
            print(f"downloading {entry['filename']} ...")
            urllib.request.urlretrieve(entry["url"], dest)
        actual = sha256_of_file(dest)
        if actual != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for {entry['filename']}: expected {entry['sha256']}, got {actual}")
        with tarfile.open(dest, "r:gz") as tar:
            _safe_extract(tar, Path(entry["extract_to"]))


def compare(name, recomputed, expected, path, mismatches):
    if isinstance(recomputed, float) and isinstance(expected, float):
        ok = abs(recomputed - expected) < 1e-9
    else:
        ok = recomputed == expected
    if not ok:
        mismatches.append(f"{path}: recomputed={recomputed!r} expected={expected!r}")


def compare_against_canonical(recomputed):
    canonical = json.loads((config.RESULTS_ROOT / "summary.json").read_text())
    mismatches = []

    if "preregistered" not in canonical:
        return mismatches  # canonical file uses a different schema wrapper; scientific fields compared below

    for family in ("H1", "H2", "H3", "BLOCK_C"):
        canon_cells = {c["run_id"]: c for c in canonical["preregistered"][family]["cells"]}
        mine_cells = recomputed["preregistered"][family]["cells"]
        for i, cell in enumerate(mine_cells):
            theirs = canon_cells[cell["run_id"]]
            for k in ("delta_accuracy", "ci_low", "ci_high", "bootstrap_seed"):
                compare(family, cell["bootstrap"][k], theirs["bootstrap"][k], f"{family}.{cell['run_id']}.bootstrap.{k}", mismatches)
            for k in ("b", "c", "n_discordant", "p_value"):
                compare(family, cell["mcnemar"][k], theirs["mcnemar"][k], f"{family}.{cell['run_id']}.mcnemar.{k}", mismatches)
        mine_mult = recomputed["preregistered"][family]["multiplicity"]["corrected_p_values"]
        canon_order = [c["run_id"] for c in canonical["preregistered"][family]["cells"]]
        canon_mult = canonical["preregistered"][family]["multiplicity"]["corrected_p_values"]
        for i, cell in enumerate(mine_cells):
            j = canon_order.index(cell["run_id"])
            compare(family, mine_mult[i], canon_mult[j], f"{family}.{cell['run_id']}.corrected_p", mismatches)

    for hyp in ("H1", "H2", "H3"):
        canon_pairs = {p["pair_id"]: p for p in canonical["secondary_cross_condition"][hyp]["pairs"]}
        for pair in recomputed["secondary_cross_condition"][hyp]["pairs"]:
            theirs = canon_pairs[pair["pair_id"]]
            for k in ("did", "ci_low", "ci_high", "bootstrap_seed"):
                compare(hyp, pair["bootstrap"][k], theirs["bootstrap"][k], f"{hyp}.{pair['pair_id']}.bootstrap.{k}", mismatches)

    return mismatches


def main():
    ensure_assets()
    recomputed = analyze.build_summary()
    mismatches = compare_against_canonical(recomputed)

    if mismatches:
        print("FAIL:", len(mismatches), "mismatch(es)")
        for m in mismatches[:50]:
            print(" -", m)
        return 1

    print("PASS: recomputed scientific summary matches committed canonical results exactly.")
    print(f"  cells checked:  {sum(recomputed['preregistered'][f]['n_cells'] for f in ('H1','H2','H3','BLOCK_C'))}")
    print(f"  pairs checked:  {sum(recomputed['secondary_cross_condition'][h]['n_pairs'] for h in ('H1','H2','H3'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

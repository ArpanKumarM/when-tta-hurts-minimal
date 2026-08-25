from pathlib import Path

DATA_ROOT = Path("assets/data")
CHECKPOINT_ROOT = Path("assets/checkpoints")
PREDICTION_ROOT = Path("assets/predictions")
RESULTS_ROOT = Path("results")

DERMAMNIST_LICENSE_NOTICE = (
    "DermaMNIST is distributed under CC BY-NC 4.0 (non-commercial use only). Source: HAM10000."
)

DATASETS = {
    "pathmnist": {
        "n_channels": 3,
        "n_classes": 9,
        "license": "CC BY 4.0",
        "splits": {"train": 89996, "val": 10004, "test": 7180},
        "medmnist_class": "PathMNIST",
        "urls": {
            28: "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
            64: "https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1",
            128: "https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1",
        },
        "md5": {
            28: "a8b06965200029087d5bd730944a56c1",
            64: "55aa9c1e0525abe5a6b9d8343a507616",
            128: "ac42d08fb904d92c244187169d1fd1d9",
        },
    },
    "bloodmnist": {
        "n_channels": 3,
        "n_classes": 8,
        "license": "CC BY 4.0",
        "splits": {"train": 11959, "val": 1712, "test": 3421},
        "medmnist_class": "BloodMNIST",
        "urls": {
            28: "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
            64: "https://zenodo.org/records/10519652/files/bloodmnist_64.npz?download=1",
            128: "https://zenodo.org/records/10519652/files/bloodmnist_128.npz?download=1",
        },
        "md5": {
            28: "7053d0359d879ad8a5505303e11de1dc",
            64: "2b94928a2ae4916078ca51e05b6b800b",
            128: "adace1e0ed228fccda1f39692059dd4c",
        },
    },
    "dermamnist": {
        "n_channels": 3,
        "n_classes": 7,
        "license": "CC BY-NC 4.0",
        "splits": {"train": 7007, "val": 1003, "test": 2005},
        "medmnist_class": "DermaMNIST",
        "urls": {
            28: "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1",
            64: "https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1",
            128: "https://zenodo.org/records/10519652/files/dermamnist_128.npz?download=1",
        },
        "md5": {
            28: "0744692d530f8e62ec473284d019b0c7",
            64: "b70a2f5635c6199aeaa28c31d7202e1f",
            128: "2defd784463fa5243564e855ed717de1",
        },
    },
}

TRAINING = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "max_epochs": 30,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
    "batch_size": 256,
}

CONFIRMATORY_SEEDS = (0, 1, 2)

# 39 preregistered matrix cells (Blocks A/B/C/D). Each dict's "attempt"
# is the canonical training-checkpoint attempt number that the final-test
# evaluation actually consumed (identical checkpoint hash to attempt 1
# for every cell except two, both resolved to attempt 2 after a
# transient, unrelated engineering retry; the trained weights are
# bit-identical to attempt 1 wherever both exist).
CELLS = [
    {"block": 'A', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-pathmnist-28px-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-pathmnist-28px-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-pathmnist-28px-batchnorm-policy-none-s2', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-pathmnist-28px-groupnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-pathmnist-28px-groupnorm-policy-none-s1', "attempt": 2},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-pathmnist-28px-groupnorm-policy-none-s2', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-pathmnist-64px-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-pathmnist-64px-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-pathmnist-64px-batchnorm-policy-none-s2', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-pathmnist-64px-groupnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-pathmnist-64px-groupnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'pathmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-pathmnist-64px-groupnorm-policy-none-s2', "attempt": 2},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s2', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s2', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s2', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 0, "run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s0', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 1, "run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s1', "attempt": 1},
    {"block": 'A', "dataset": 'bloodmnist', "resolution": 64, "model": 'small_cnn', "normalization": 'groupnorm', "training_policy": 'none', "seed": 2, "run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s2', "attempt": 1},
    {"block": 'B', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'matched_to_approved_tta_policy', "seed": 0, "run_id": 'B-pathmnist-28px-batchnorm-policy-matched_mixed-s0', "attempt": 1},
    {"block": 'B', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'matched_to_approved_tta_policy', "seed": 1, "run_id": 'B-pathmnist-28px-batchnorm-policy-matched_mixed-s1', "attempt": 1},
    {"block": 'B', "dataset": 'pathmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'matched_to_approved_tta_policy', "seed": 2, "run_id": 'B-pathmnist-28px-batchnorm-policy-matched_mixed-s2', "attempt": 1},
    {"block": 'B', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'matched_to_approved_tta_policy', "seed": 0, "run_id": 'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0', "attempt": 1},
    {"block": 'B', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'matched_to_approved_tta_policy', "seed": 1, "run_id": 'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1', "attempt": 1},
    {"block": 'B', "dataset": 'bloodmnist', "resolution": 28, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'matched_to_approved_tta_policy', "seed": 2, "run_id": 'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2', "attempt": 1},
    {"block": 'C', "dataset": 'dermamnist', "resolution": 28, "model": 'resnet18', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'C-dermamnist-28px-resnet18-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'C', "dataset": 'dermamnist', "resolution": 28, "model": 'resnet18', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'C-dermamnist-28px-resnet18-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'C', "dataset": 'dermamnist', "resolution": 28, "model": 'resnet18', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'C-dermamnist-28px-resnet18-batchnorm-policy-none-s2', "attempt": 1},
    {"block": 'D', "dataset": 'pathmnist', "resolution": 128, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'D-pathmnist-128px-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'D', "dataset": 'pathmnist', "resolution": 128, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'D-pathmnist-128px-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'D', "dataset": 'pathmnist', "resolution": 128, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'D-pathmnist-128px-batchnorm-policy-none-s2', "attempt": 1},
    {"block": 'D', "dataset": 'bloodmnist', "resolution": 128, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 0, "run_id": 'D-bloodmnist-128px-batchnorm-policy-none-s0', "attempt": 1},
    {"block": 'D', "dataset": 'bloodmnist', "resolution": 128, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 1, "run_id": 'D-bloodmnist-128px-batchnorm-policy-none-s1', "attempt": 1},
    {"block": 'D', "dataset": 'bloodmnist', "resolution": 128, "model": 'small_cnn', "normalization": 'batchnorm', "training_policy": 'none', "seed": 2, "run_id": 'D-bloodmnist-128px-batchnorm-policy-none-s2', "attempt": 1},
]

TTA_SEED = 1306178015
PREFIX_SEQUENCE = (1, 2, 5, 10, 25, 50, 100)
MAX_VIEWS = 100
PRIMARY_N = 50
PRIMARY_AGGREGATION = "mean_probability"
AGGREGATORS = ("mean_probability", "majority_vote", "confidence_weighted_average")
POLICY_IDENTIFIER = "mixed"
INFERENCE_BATCH_SIZE = 256
BN_ADAPTATION_BATCH_SIZE = 256

BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_CI_LEVEL = 0.95

# Exact bootstrap seeds used to produce the canonical results, keyed by
# (run_id -> family -> seed) for the preregistered within-cell analysis.
# A cell shared by more than one hypothesis family gets a different seed
# per family (each family's analysis was sealed independently), matching
# the canonical generation-2 evidence exactly.
CELL_BOOTSTRAP_SEEDS = {
    'A-bloodmnist-28px-batchnorm-policy-none-s0': {'H1': 6852822931899801001, 'H2': 14914118073286518053, 'H3': 1881712587196421808},
    'A-bloodmnist-28px-batchnorm-policy-none-s1': {'H1': 107582494495540244, 'H2': 4390784821866665153, 'H3': 17236453188950683580},
    'A-bloodmnist-28px-batchnorm-policy-none-s2': {'H1': 17726481440799308212, 'H2': 2780011367248908426, 'H3': 4278892122902357225},
    'A-bloodmnist-28px-groupnorm-policy-none-s0': {'H1': 12750801987029547776, 'H2': 6324267604844760021},
    'A-bloodmnist-28px-groupnorm-policy-none-s1': {'H1': 235032471045393434, 'H2': 7952380049961335666},
    'A-bloodmnist-28px-groupnorm-policy-none-s2': {'H1': 4574417551187121806, 'H2': 12865449219485634108},
    'A-bloodmnist-64px-batchnorm-policy-none-s0': {'H1': 11357616583560382298, 'H2': 2472922441272587476},
    'A-bloodmnist-64px-batchnorm-policy-none-s1': {'H1': 9378878904874818530, 'H2': 10765801285216349956},
    'A-bloodmnist-64px-batchnorm-policy-none-s2': {'H1': 2514762302957593549, 'H2': 15262229346769325271},
    'A-bloodmnist-64px-groupnorm-policy-none-s0': {'H1': 4595344248825187966, 'H2': 15620535269584933612},
    'A-bloodmnist-64px-groupnorm-policy-none-s1': {'H1': 17395529015818933078, 'H2': 2013624468267925638},
    'A-bloodmnist-64px-groupnorm-policy-none-s2': {'H1': 3666153176460435593, 'H2': 17807372843321106994},
    'A-pathmnist-28px-batchnorm-policy-none-s0': {'H1': 16668554067633858434, 'H2': 9200543376725575507, 'H3': 15427141989847823402},
    'A-pathmnist-28px-batchnorm-policy-none-s1': {'H1': 61642027187643881, 'H2': 12844637529964701630, 'H3': 2965128329117413509},
    'A-pathmnist-28px-batchnorm-policy-none-s2': {'H1': 16194590597561182102, 'H2': 8542282690091124942, 'H3': 6283577493862277125},
    'A-pathmnist-28px-groupnorm-policy-none-s0': {'H1': 3589027546004228221, 'H2': 2263624867259540459},
    'A-pathmnist-28px-groupnorm-policy-none-s1': {'H1': 10608172696847971723, 'H2': 11132852760136157363},
    'A-pathmnist-28px-groupnorm-policy-none-s2': {'H1': 9282842821468484875, 'H2': 14428403398067458886},
    'A-pathmnist-64px-batchnorm-policy-none-s0': {'H1': 13867583005211849326, 'H2': 8345236936397340051},
    'A-pathmnist-64px-batchnorm-policy-none-s1': {'H1': 9809587394341235416, 'H2': 15351922106192118843},
    'A-pathmnist-64px-batchnorm-policy-none-s2': {'H1': 14810902128797875505, 'H2': 17619663637018341732},
    'A-pathmnist-64px-groupnorm-policy-none-s0': {'H1': 5607275821957592149, 'H2': 13147060227191197776},
    'A-pathmnist-64px-groupnorm-policy-none-s1': {'H1': 13714144206001147446, 'H2': 16264158754045271371},
    'A-pathmnist-64px-groupnorm-policy-none-s2': {'H1': 13852523348635621644, 'H2': 13162503019442811933},
    'D-bloodmnist-128px-batchnorm-policy-none-s0': {'H2': 11223917528265805721},
    'D-bloodmnist-128px-batchnorm-policy-none-s1': {'H2': 16071611033684289982},
    'D-bloodmnist-128px-batchnorm-policy-none-s2': {'H2': 3314451222932862449},
    'D-pathmnist-128px-batchnorm-policy-none-s0': {'H2': 10696692979006334199},
    'D-pathmnist-128px-batchnorm-policy-none-s1': {'H2': 4687827889396054682},
    'D-pathmnist-128px-batchnorm-policy-none-s2': {'H2': 10338336235593698305},
    'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0': {'H3': 17149266500120055747},
    'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1': {'H3': 15068239698090101854},
    'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2': {'H3': 885011341447096431},
    'B-pathmnist-28px-batchnorm-policy-matched_mixed-s0': {'H3': 5033340222691814508},
    'B-pathmnist-28px-batchnorm-policy-matched_mixed-s1': {'H3': 3361683170017310573},
    'B-pathmnist-28px-batchnorm-policy-matched_mixed-s2': {'H3': 9433477654027153965},
    'C-dermamnist-28px-resnet18-batchnorm-policy-none-s0': {'BLOCK_C': 7702874108069711248},
    'C-dermamnist-28px-resnet18-batchnorm-policy-none-s1': {'BLOCK_C': 18423129121683055617},
    'C-dermamnist-28px-resnet18-batchnorm-policy-none-s2': {'BLOCK_C': 204995360263658322},
}

# 30 secondary fixed-model DiD pairs (H1: normalization, H2: resolution,
# H3: policy matching), each with the exact bootstrap seed used to
# produce the canonical secondary results.
PAIRS = [
    {"hypothesis": 'H1', "pair_id": 'H1-bloodmnist-28px-s0', "dataset": 'bloodmnist', "seed": 0, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s0', "condition_b_run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s0'},
    {"hypothesis": 'H1', "pair_id": 'H1-bloodmnist-28px-s1', "dataset": 'bloodmnist', "seed": 1, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s1', "condition_b_run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s1'},
    {"hypothesis": 'H1', "pair_id": 'H1-bloodmnist-28px-s2', "dataset": 'bloodmnist', "seed": 2, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s2', "condition_b_run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s2'},
    {"hypothesis": 'H1', "pair_id": 'H1-bloodmnist-64px-s0', "dataset": 'bloodmnist', "seed": 0, "condition_a_run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s0', "condition_b_run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s0'},
    {"hypothesis": 'H1', "pair_id": 'H1-bloodmnist-64px-s1', "dataset": 'bloodmnist', "seed": 1, "condition_a_run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s1', "condition_b_run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s1'},
    {"hypothesis": 'H1', "pair_id": 'H1-bloodmnist-64px-s2', "dataset": 'bloodmnist', "seed": 2, "condition_a_run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s2', "condition_b_run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s2'},
    {"hypothesis": 'H1', "pair_id": 'H1-pathmnist-28px-s0', "dataset": 'pathmnist', "seed": 0, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s0', "condition_b_run_id": 'A-pathmnist-28px-groupnorm-policy-none-s0'},
    {"hypothesis": 'H1', "pair_id": 'H1-pathmnist-28px-s1', "dataset": 'pathmnist', "seed": 1, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s1', "condition_b_run_id": 'A-pathmnist-28px-groupnorm-policy-none-s1'},
    {"hypothesis": 'H1', "pair_id": 'H1-pathmnist-28px-s2', "dataset": 'pathmnist', "seed": 2, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s2', "condition_b_run_id": 'A-pathmnist-28px-groupnorm-policy-none-s2'},
    {"hypothesis": 'H1', "pair_id": 'H1-pathmnist-64px-s0', "dataset": 'pathmnist', "seed": 0, "condition_a_run_id": 'A-pathmnist-64px-batchnorm-policy-none-s0', "condition_b_run_id": 'A-pathmnist-64px-groupnorm-policy-none-s0'},
    {"hypothesis": 'H1', "pair_id": 'H1-pathmnist-64px-s1', "dataset": 'pathmnist', "seed": 1, "condition_a_run_id": 'A-pathmnist-64px-batchnorm-policy-none-s1', "condition_b_run_id": 'A-pathmnist-64px-groupnorm-policy-none-s1'},
    {"hypothesis": 'H1', "pair_id": 'H1-pathmnist-64px-s2', "dataset": 'pathmnist', "seed": 2, "condition_a_run_id": 'A-pathmnist-64px-batchnorm-policy-none-s2', "condition_b_run_id": 'A-pathmnist-64px-groupnorm-policy-none-s2'},
    {"hypothesis": 'H2', "pair_id": 'H2-bloodmnist-batchnorm-s0', "dataset": 'bloodmnist', "seed": 0, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s0', "condition_b_run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s0'},
    {"hypothesis": 'H2', "pair_id": 'H2-bloodmnist-batchnorm-s1', "dataset": 'bloodmnist', "seed": 1, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s1', "condition_b_run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s1'},
    {"hypothesis": 'H2', "pair_id": 'H2-bloodmnist-batchnorm-s2', "dataset": 'bloodmnist', "seed": 2, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s2', "condition_b_run_id": 'A-bloodmnist-64px-batchnorm-policy-none-s2'},
    {"hypothesis": 'H2', "pair_id": 'H2-bloodmnist-groupnorm-s0', "dataset": 'bloodmnist', "seed": 0, "condition_a_run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s0', "condition_b_run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s0'},
    {"hypothesis": 'H2', "pair_id": 'H2-bloodmnist-groupnorm-s1', "dataset": 'bloodmnist', "seed": 1, "condition_a_run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s1', "condition_b_run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s1'},
    {"hypothesis": 'H2', "pair_id": 'H2-bloodmnist-groupnorm-s2', "dataset": 'bloodmnist', "seed": 2, "condition_a_run_id": 'A-bloodmnist-28px-groupnorm-policy-none-s2', "condition_b_run_id": 'A-bloodmnist-64px-groupnorm-policy-none-s2'},
    {"hypothesis": 'H2', "pair_id": 'H2-pathmnist-batchnorm-s0', "dataset": 'pathmnist', "seed": 0, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s0', "condition_b_run_id": 'A-pathmnist-64px-batchnorm-policy-none-s0'},
    {"hypothesis": 'H2', "pair_id": 'H2-pathmnist-batchnorm-s1', "dataset": 'pathmnist', "seed": 1, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s1', "condition_b_run_id": 'A-pathmnist-64px-batchnorm-policy-none-s1'},
    {"hypothesis": 'H2', "pair_id": 'H2-pathmnist-batchnorm-s2', "dataset": 'pathmnist', "seed": 2, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s2', "condition_b_run_id": 'A-pathmnist-64px-batchnorm-policy-none-s2'},
    {"hypothesis": 'H2', "pair_id": 'H2-pathmnist-groupnorm-s0', "dataset": 'pathmnist', "seed": 0, "condition_a_run_id": 'A-pathmnist-28px-groupnorm-policy-none-s0', "condition_b_run_id": 'A-pathmnist-64px-groupnorm-policy-none-s0'},
    {"hypothesis": 'H2', "pair_id": 'H2-pathmnist-groupnorm-s1', "dataset": 'pathmnist', "seed": 1, "condition_a_run_id": 'A-pathmnist-28px-groupnorm-policy-none-s1', "condition_b_run_id": 'A-pathmnist-64px-groupnorm-policy-none-s1'},
    {"hypothesis": 'H2', "pair_id": 'H2-pathmnist-groupnorm-s2', "dataset": 'pathmnist', "seed": 2, "condition_a_run_id": 'A-pathmnist-28px-groupnorm-policy-none-s2', "condition_b_run_id": 'A-pathmnist-64px-groupnorm-policy-none-s2'},
    {"hypothesis": 'H3', "pair_id": 'H3-bloodmnist-s0', "dataset": 'bloodmnist', "seed": 0, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s0', "condition_b_run_id": 'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0'},
    {"hypothesis": 'H3', "pair_id": 'H3-bloodmnist-s1', "dataset": 'bloodmnist', "seed": 1, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s1', "condition_b_run_id": 'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1'},
    {"hypothesis": 'H3', "pair_id": 'H3-bloodmnist-s2', "dataset": 'bloodmnist', "seed": 2, "condition_a_run_id": 'A-bloodmnist-28px-batchnorm-policy-none-s2', "condition_b_run_id": 'B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2'},
    {"hypothesis": 'H3', "pair_id": 'H3-pathmnist-s0', "dataset": 'pathmnist', "seed": 0, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s0', "condition_b_run_id": 'B-pathmnist-28px-batchnorm-policy-matched_mixed-s0'},
    {"hypothesis": 'H3', "pair_id": 'H3-pathmnist-s1', "dataset": 'pathmnist', "seed": 1, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s1', "condition_b_run_id": 'B-pathmnist-28px-batchnorm-policy-matched_mixed-s1'},
    {"hypothesis": 'H3', "pair_id": 'H3-pathmnist-s2', "dataset": 'pathmnist', "seed": 2, "condition_a_run_id": 'A-pathmnist-28px-batchnorm-policy-none-s2', "condition_b_run_id": 'B-pathmnist-28px-batchnorm-policy-matched_mixed-s2'},
]

PAIR_BOOTSTRAP_SEEDS = {
    'H1-bloodmnist-28px-s0': 4390020174277029474,
    'H1-bloodmnist-28px-s1': 6504871812854538437,
    'H1-bloodmnist-28px-s2': 16357197268626080736,
    'H1-bloodmnist-64px-s0': 16030802730709054075,
    'H1-bloodmnist-64px-s1': 1022290854441265171,
    'H1-bloodmnist-64px-s2': 3368519047823951539,
    'H1-pathmnist-28px-s0': 2984921535713317300,
    'H1-pathmnist-28px-s1': 18393649286561710106,
    'H1-pathmnist-28px-s2': 3172475415844935014,
    'H1-pathmnist-64px-s0': 4216938225344028506,
    'H1-pathmnist-64px-s1': 14727784725396732187,
    'H1-pathmnist-64px-s2': 1119846549630843117,
    'H2-bloodmnist-batchnorm-s0': 4979525161427500434,
    'H2-bloodmnist-batchnorm-s1': 14457666715197793182,
    'H2-bloodmnist-batchnorm-s2': 13081380704952742258,
    'H2-bloodmnist-groupnorm-s0': 8802557636938842384,
    'H2-bloodmnist-groupnorm-s1': 4110850257161389096,
    'H2-bloodmnist-groupnorm-s2': 5868921605707706794,
    'H2-pathmnist-batchnorm-s0': 5736386768345762835,
    'H2-pathmnist-batchnorm-s1': 6685182847633340356,
    'H2-pathmnist-batchnorm-s2': 5859008337064815893,
    'H2-pathmnist-groupnorm-s0': 3849713377287158877,
    'H2-pathmnist-groupnorm-s1': 9036897613002034948,
    'H2-pathmnist-groupnorm-s2': 17646053230732506305,
    'H3-bloodmnist-s0': 9046024898869167128,
    'H3-bloodmnist-s1': 14470856255113813463,
    'H3-bloodmnist-s2': 13268576253975265310,
    'H3-pathmnist-s0': 1810779066829526408,
    'H3-pathmnist-s1': 8507674302219064030,
    'H3-pathmnist-s2': 3127448399717998130,
}

BLOCK_C_EXTERNAL_REFERENCE_PP = 1.6

from pathlib import Path

ENV_ID = "FrozenLake-v1"
MAP = "8x8"
IS_SLIDE = True

CURVE_WINDOW_SIZE = 500
TRAINING_TIMES = 50000
EVALUATE_TIMES = 500
MULTI_SEEDS = (42, 123, 2026)
EVALUATE_SEED_OFFSET = 10000
ALGORITHMS = ("q_learning", "sarsa")
SLIPPERY_OPTIONS = (False, True)

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_EPISODES = 40000
if EPSILON_START == EPSILON_END:
    EPSILON_SCHEDULE_TYPE = "fixed"
    EXPERIMENT_TAG = f"fixed_epsilon_{EPSILON_START:g}"
else:
    EPSILON_SCHEDULE_TYPE = "linear"
    EXPERIMENT_TAG = (
        f"linear_epsilon_{EPSILON_START:g}"
        f"_to_{EPSILON_END:g}"
        f"_over_{EPSILON_DECAY_EPISODES}"
    )

ALPHA = 0.1
GAMMA = 0.99
MAX_STEP = 200
TRAIN_SEED = 42
EVALUATE_SEED = TRAIN_SEED + EVALUATE_SEED_OFFSET
ALGORITHM = "sarsa"
ENV_MODE = "slippery" if IS_SLIDE else "deterministic"


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def build_run_dir(
    algorithm,
    is_slippery,
    train_seed,
    map_name=MAP,
    experiment_tag=EXPERIMENT_TAG,
):
    environment_mode = "slippery" if is_slippery else "deterministic"
    return (
        OUTPUT_DIR
        / algorithm
        / f"{map_name}_{environment_mode}"
        / experiment_tag
        / f"seed_{train_seed}"
    )


RUN_DIR = build_run_dir(ALGORITHM, IS_SLIDE, TRAIN_SEED)
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
Q_TABLE_PATH = CHECKPOINT_DIR / "q_table.npy"
CONFIG_PATH = RUN_DIR / "config.json"
METRICS_PATH = RUN_DIR / "metrics.json"
COMPARISON_DIR = (
    OUTPUT_DIR
    / "comparisons"
    / f"{MAP}_q_learning_vs_sarsa"
    / EXPERIMENT_TAG
)
MULTISEED_RUNS_PATH = COMPARISON_DIR / "runs.json"
MULTISEED_SUMMARY_PATH = COMPARISON_DIR / "summary.json"
COMPARISON_CURVE_PATH = (
    COMPARISON_DIR / "curves" / "evaluation_success_rate.png"
)


EXPERIMENT_CONFIG = {
    "experiment_tag": EXPERIMENT_TAG,
    "algorithm": ALGORITHM,
    "environment": {
        "env_id": ENV_ID,
        "map_name": MAP,
        "is_slippery": IS_SLIDE,
    },
    "training": {
        "epsilon_schedule": {
            "type": EPSILON_SCHEDULE_TYPE,
            "start": EPSILON_START,
            "end": EPSILON_END,
            "decay_episodes": EPSILON_DECAY_EPISODES,
        },
        "num_episodes": TRAINING_TIMES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "max_steps": MAX_STEP,
        "seed": TRAIN_SEED,
    },
    "evaluation": {
        "num_episodes": EVALUATE_TIMES,
        "max_steps": MAX_STEP,
        "epsilon": 0.0,
        "seed": EVALUATE_SEED,
    },
    "visualization": {
        "training_curve_window_size": CURVE_WINDOW_SIZE,
    },
}





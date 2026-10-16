import numpy as np

import artifacts
import config
import experiment


SUMMARY_METRICS = {
    "training_last_500_success_rate": (
        "training",
        "last_500_success_rate",
    ),
    "evaluation_success_rate": ("evaluation", "success_rate"),
    "evaluation_average_reward": ("evaluation", "average_reward"),
    "evaluation_average_steps": ("evaluation", "average_steps"),
    "evaluation_successful_average_steps": (
        "evaluation",
        "successful_average_steps",
    ),
}


def summarize_values(values):
    if not values:
        return {
            "values": [],
            "mean": None,
            "sample_std": None,
        }

    numeric_values = np.asarray(values, dtype=float)
    sample_std = (
        float(np.std(numeric_values, ddof=1))
        if len(numeric_values) > 1
        else 0.0
    )
    return {
        "values": [float(value) for value in numeric_values],
        "mean": float(np.mean(numeric_values)),
        "sample_std": sample_std,
    }


def summarize_runs(runs):
    groups = []
    for is_slippery in config.SLIPPERY_OPTIONS:
        environment_mode = "slippery" if is_slippery else "deterministic"
        for algorithm in config.ALGORITHMS:
            group_runs = [
                run
                for run in runs
                if run["algorithm"] == algorithm
                and run["is_slippery"] == is_slippery
            ]
            metrics = {}
            for output_name, (section, metric_name) in SUMMARY_METRICS.items():
                values = [
                    run["metrics"][section][metric_name]
                    for run in group_runs
                    if run["metrics"][section][metric_name] is not None
                ]
                metrics[output_name] = summarize_values(values)

            groups.append(
                {
                    "algorithm": algorithm,
                    "environment_mode": environment_mode,
                    "is_slippery": is_slippery,
                    "num_seeds": len(group_runs),
                    "train_seeds": [run["train_seed"] for run in group_runs],
                    "metrics": metrics,
                }
            )

    return {
        "experiment_tag": config.EXPERIMENT_TAG,
        "map_name": config.MAP,
        "sample_std_ddof": 1,
        "groups": groups,
    }


def run_all_experiments():
    runs = []
    for is_slippery in config.SLIPPERY_OPTIONS:
        for algorithm in config.ALGORITHMS:
            for train_seed in config.MULTI_SEEDS:
                evaluate_seed = train_seed + config.EVALUATE_SEED_OFFSET
                result = experiment.run_experiment(
                    algorithm=algorithm,
                    is_slippery=is_slippery,
                    train_seed=train_seed,
                    evaluate_seed=evaluate_seed,
                    map_name=config.MAP,
                )
                runs.append(result)
                print(
                    "completed:",
                    result["environment_mode"],
                    algorithm,
                    f"seed={train_seed}",
                    "success_rate=",
                    result["metrics"]["evaluation"]["success_rate"],
                )

    runs_report = {
        "experiment_tag": config.EXPERIMENT_TAG,
        "map_name": config.MAP,
        "train_seeds": list(config.MULTI_SEEDS),
        "evaluate_seed_offset": config.EVALUATE_SEED_OFFSET,
        "runs": runs,
    }
    summary_report = summarize_runs(runs)
    artifacts.save_json(runs_report, config.MULTISEED_RUNS_PATH)
    artifacts.save_json(summary_report, config.MULTISEED_SUMMARY_PATH)
    return runs_report, summary_report


def print_summary(summary_report):
    print("\nsummary: mean +/- sample_std")
    for group in summary_report["groups"]:
        success = group["metrics"]["evaluation_success_rate"]
        steps = group["metrics"]["evaluation_average_steps"]
        print(
            group["environment_mode"],
            group["algorithm"],
            f"success={success['mean']:.3f} +/- {success['sample_std']:.3f}",
            f"steps={steps['mean']:.3f} +/- {steps['sample_std']:.3f}",
        )


def main():
    _, summary_report = run_all_experiments()
    print_summary(summary_report)
    print("runs_saved_to:", config.MULTISEED_RUNS_PATH)
    print("summary_saved_to:", config.MULTISEED_SUMMARY_PATH)


if __name__ == "__main__":
    main()

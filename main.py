import config
import experiment


def main():
    result = experiment.run_experiment(
        algorithm=config.ALGORITHM,
        is_slippery=config.IS_SLIDE,
        train_seed=config.TRAIN_SEED,
        evaluate_seed=config.EVALUATE_SEED,
        map_name=config.MAP,
    )
    metrics = result["metrics"]

    print("q_table_saved_to:", result["q_table_path"])
    print("config_saved_to:", result["config_path"])
    print("metrics_saved_to:", result["metrics_path"])
    print(
        "last_500_train_success_rate:",
        metrics["training"]["last_500_success_rate"],
    )
    print(
        "evaluation_success_rate:",
        metrics["evaluation"]["success_rate"],
    )
    print(
        "evaluation_average_steps:",
        metrics["evaluation"]["average_steps"],
    )
    print(
        "training_curve_saved_to:",
        result["training_curve_path"],
    )

if __name__ == "__main__":
    main()

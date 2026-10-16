import config


def test_experiment_config_matches_runtime_constants():
    expected_mode = "slippery" if config.IS_SLIDE else "deterministic"
    expected_run_dir = config.build_run_dir(
        config.ALGORITHM,
        config.IS_SLIDE,
        config.TRAIN_SEED,
    )

    assert config.ENV_MODE == expected_mode
    assert config.RUN_DIR == expected_run_dir
    assert config.CHECKPOINT_DIR == expected_run_dir / "checkpoints"
    assert config.Q_TABLE_PATH == expected_run_dir / "checkpoints" / "q_table.npy"
    assert config.CONFIG_PATH == expected_run_dir / "config.json"
    assert config.METRICS_PATH == expected_run_dir / "metrics.json"
    assert config.EXPERIMENT_CONFIG == {
        "experiment_tag": config.EXPERIMENT_TAG,
        "algorithm": config.ALGORITHM,
        "environment": {
            "env_id": config.ENV_ID,
            "map_name": config.MAP,
            "is_slippery": config.IS_SLIDE,
        },
        "training": {
            "num_episodes": config.TRAINING_TIMES,
            "epsilon_schedule": {
                "type": config.EPSILON_SCHEDULE_TYPE,
                "start": config.EPSILON_START,
                "end": config.EPSILON_END,
                "decay_episodes": config.EPSILON_DECAY_EPISODES,
            },
            "alpha": config.ALPHA,
            "gamma": config.GAMMA,
            "max_steps": config.MAX_STEP,
            "seed": config.TRAIN_SEED,
        },
        "evaluation": {
            "num_episodes": config.EVALUATE_TIMES,
            "max_steps": config.MAX_STEP,
            "epsilon": 0.0,
            "seed": config.EVALUATE_SEED,
        },
        "visualization": {
            "training_curve_window_size": config.CURVE_WINDOW_SIZE,
        },
    }


def test_build_run_dir_isolates_algorithm_environment_and_seed():
    run_dir = config.build_run_dir(
        algorithm="q_learning",
        is_slippery=False,
        train_seed=123,
    )

    assert run_dir == (
        config.OUTPUT_DIR
        / "q_learning"
        / f"{config.MAP}_deterministic"
        / config.EXPERIMENT_TAG
        / "seed_123"
    )


def test_multiseed_comparison_paths_use_the_experiment_tag():
    expected_dir = (
        config.OUTPUT_DIR
        / "comparisons"
        / f"{config.MAP}_q_learning_vs_sarsa"
        / config.EXPERIMENT_TAG
    )

    assert config.MULTI_SEEDS == (42, 123, 2026)
    assert config.COMPARISON_DIR == expected_dir
    assert config.MULTISEED_RUNS_PATH == expected_dir / "runs.json"
    assert config.MULTISEED_SUMMARY_PATH == expected_dir / "summary.json"
    assert config.COMPARISON_CURVE_PATH == (
        expected_dir / "curves" / "evaluation_success_rate.png"
    )

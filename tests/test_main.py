import main


def test_main_runs_the_configured_experiment_and_prints_results(
    monkeypatch,
    capsys,
):
    call = {}

    def fake_run_experiment(**kwargs):
        call.update(kwargs)
        return {
            "q_table_path": "q_table.npy",
            "config_path": "config.json",
            "metrics_path": "metrics.json",
            "training_curve_path": "training_success_rate.png",
            "metrics": {
                "training": {"last_500_success_rate": 0.8},
                "evaluation": {
                    "success_rate": 0.7,
                    "average_steps": 12.5,
                },
            },
        }

    monkeypatch.setattr(main.experiment, "run_experiment", fake_run_experiment)

    main.main()

    assert call == {
        "algorithm": main.config.ALGORITHM,
        "is_slippery": main.config.IS_SLIDE,
        "train_seed": main.config.TRAIN_SEED,
        "evaluate_seed": main.config.EVALUATE_SEED,
        "map_name": main.config.MAP,
    }
    output = capsys.readouterr().out
    assert "q_table_saved_to: q_table.npy" in output
    assert "config_saved_to: config.json" in output
    assert "metrics_saved_to: metrics.json" in output
    assert "training_curve_saved_to: training_success_rate.png" in output
    assert "last_500_train_success_rate: 0.8" in output
    assert "evaluation_success_rate: 0.7" in output
    assert "evaluation_average_steps: 12.5" in output

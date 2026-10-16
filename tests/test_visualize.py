import numpy as np
import pytest

import visualize


def test_moving_average_calculates_each_complete_window():
    result = visualize.moving_average([0, 1, 0, 1], window_size=2)

    np.testing.assert_allclose(result, [0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (1, [1.0, 2.0, 3.0]),
        (3, [2.0]),
    ],
)
def test_moving_average_supports_boundary_window_sizes(window_size, expected):
    result = visualize.moving_average([1, 2, 3], window_size)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("window_size", [0, -1, 4])
def test_moving_average_rejects_invalid_window_sizes(window_size):
    with pytest.raises(ValueError):
        visualize.moving_average([1, 2, 3], window_size)


def test_moving_average_does_not_modify_input():
    values = np.array([0.0, 1.0, 0.0, 1.0])
    original = values.copy()

    visualize.moving_average(values, window_size=2)

    np.testing.assert_array_equal(values, original)


def test_plot_training_curve_creates_png_and_closes_figure(tmp_path):
    output_path = tmp_path / "nested" / "training_curve.png"
    history = {"rewards": [0, 1, 0, 1, 1, 1]}

    result = visualize.plot_training_curve(
        history,
        output_path,
        window_size=3,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert visualize.plt.get_fignums() == []


def test_plot_evaluation_comparison_creates_png_and_closes_figure(tmp_path):
    summary_report = {
        "groups": [
            {
                "environment_mode": environment_mode,
                "algorithm": algorithm,
                "metrics": {
                    "evaluation_success_rate": {
                        "mean": mean,
                        "sample_std": sample_std,
                    }
                },
            }
            for environment_mode, algorithm, mean, sample_std in [
                ("deterministic", "q_learning", 1.0, 0.0),
                ("deterministic", "sarsa", 1.0, 0.0),
                ("slippery", "q_learning", 0.6, 0.02),
                ("slippery", "sarsa", 0.5, 0.15),
            ]
        ]
    }
    output_path = tmp_path / "curves" / "evaluation_comparison.png"

    result = visualize.plot_evaluation_comparison(
        summary_report,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert visualize.plt.get_fignums() == []

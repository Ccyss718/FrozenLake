from pathlib import Path

import plot_results


def test_generate_comparison_plot_loads_summary_without_training(monkeypatch):
    summary_path = Path("summary.json")
    output_path = Path("curves/evaluation_success_rate.png")
    summary_report = {"groups": []}
    calls = {}

    def fake_load_json(path):
        calls["summary_path"] = path
        return summary_report

    def fake_plot(summary, path):
        calls["summary"] = summary
        calls["output_path"] = path
        return path

    monkeypatch.setattr(plot_results.artifacts, "load_json", fake_load_json)
    monkeypatch.setattr(
        plot_results.visualize,
        "plot_evaluation_comparison",
        fake_plot,
    )

    result = plot_results.generate_comparison_plot(
        summary_path,
        output_path,
    )

    assert calls == {
        "summary_path": summary_path,
        "summary": summary_report,
        "output_path": output_path,
    }
    assert result == output_path


def test_main_prints_the_saved_comparison_path(monkeypatch, capsys):
    output_path = Path("curves/evaluation_success_rate.png")
    monkeypatch.setattr(
        plot_results,
        "generate_comparison_plot",
        lambda: output_path,
    )

    plot_results.main()

    assert (
        "comparison_curve_saved_to: "
        "curves\\evaluation_success_rate.png"
    ) in capsys.readouterr().out

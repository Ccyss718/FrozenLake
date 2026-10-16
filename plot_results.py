import artifacts
import config
import visualize


def generate_comparison_plot(
    summary_path=config.MULTISEED_SUMMARY_PATH,
    output_path=config.COMPARISON_CURVE_PATH,
):
    summary_report = artifacts.load_json(summary_path)
    return visualize.plot_evaluation_comparison(
        summary_report,
        output_path,
    )


def main():
    output_path = generate_comparison_plot()
    print("comparison_curve_saved_to:", output_path)


if __name__ == "__main__":
    main()

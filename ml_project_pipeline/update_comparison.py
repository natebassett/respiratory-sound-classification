# update_comparison.py

from visualisations import (
    create_combined_results_csv,
    plot_classical_vs_cnn,
)


def main():

    print("Updating combined results...")

    create_combined_results_csv(
        classical_results_csv="results/model_results.csv",
        cnn_results_csv="results/cnn_results.csv",
    )

    print("Generating comparison graph...")

    plot_classical_vs_cnn()

    print("\nComparison outputs updated successfully.")


if __name__ == "__main__":
    main()
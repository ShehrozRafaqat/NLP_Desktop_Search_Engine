"""Run all prompt engineering experiments and generate figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.assignment_core import load_results_bundle, run_all_experiments


def _build_parameter_figure(base_dir: Path) -> None:
    df = pd.read_csv(base_dir / "results" / "parameter_sensitivity.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    score_cols = ["coherence", "creativity", "repetition", "topic_drift"]
    df.set_index("experiment_id")[score_cols].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Part 1: Parameter Sensitivity Scores")
    axes[0].set_xlabel("Experiment")
    axes[0].set_ylabel("Score (1-5)")
    axes[0].legend(loc="upper right", ncols=4, fontsize=8)

    df.plot(
        x="experiment_id",
        y=["word_count", "repetition_ratio"],
        kind="bar",
        ax=axes[1],
        color=["#2a9d8f", "#e76f51"],
    )
    axes[1].set_title("Output Length and Repetition Ratio")
    axes[1].set_xlabel("Experiment")
    axes[1].set_ylabel("Value")

    figure_path = base_dir / "figures" / "parameter_sensitivity.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def _build_prompt_figure(base_dir: Path) -> None:
    df = pd.read_csv(base_dir / "results" / "prompt_optimization.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    score_cols = ["clarity", "structure", "completeness", "factual_accuracy"]
    df.set_index("technique")[score_cols].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Part 2: Prompt Technique Comparison")
    axes[0].set_xlabel("Technique")
    axes[0].set_ylabel("Score (1-5)")
    axes[0].legend(loc="upper right", ncols=4, fontsize=8)

    df.plot(
        x="technique",
        y=["word_count"],
        kind="bar",
        ax=axes[1],
        color=["#264653"],
        legend=False,
    )
    axes[1].set_title("Response Length by Prompt Technique")
    axes[1].set_xlabel("Technique")
    axes[1].set_ylabel("Word count")

    figure_path = base_dir / "figures" / "prompt_optimization.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prompt engineering assignment workflow")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="Prompt assignment base directory",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached results if available instead of calling the API again",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    run_all_experiments(base_dir, use_cache=args.use_cache)
    load_results_bundle(base_dir)
    _build_parameter_figure(base_dir)
    _build_prompt_figure(base_dir)
    print(f"Results and figures generated under {base_dir}")


if __name__ == "__main__":
    main()

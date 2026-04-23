from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).with_name("results")
SUMMARY_OUTPUT = RESULTS_DIR / "combined_tokenizer_dataset_summary.csv"
WINNER_OUTPUT = RESULTS_DIR / "combined_tokenizer_winner_rates.csv"
CHART_OUTPUT = RESULTS_DIR / "combined_tokenizer_dataset_comparison.png"

DATASET_FILES = {
    "Wikipedia": "100_wikipedia_avg.csv",
    "Common Crawl": "common_crawl_tokenizer_comparison_record_level_results.csv",
    "OpenSubtitles": "opensubtitles_tokenizer_comparison_record_level_results.csv",
}

TOKENIZER_ORDER = ["gpt2", "llama-sp", "t5", "wordpiece"]
DATASET_ORDER = list(DATASET_FILES)
PALETTE = {
    "gpt2": "#4C78A8",
    "llama-sp": "#F58518",
    "t5": "#54A24B",
    "wordpiece": "#B279A2",
}

METRICS = [
    "token_count",
    "compression_ratio",
    "oov_rate",
    "fertility_rate",
    "mean_tokens_per_sentence",
    "pieces_per_word",
    "max_pieces_per_word",
    "estimated_cost",
    "nsl",
]


def load_record_level_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for dataset_name, file_name in DATASET_FILES.items():
        path = RESULTS_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")

        data = pd.read_csv(path)
        data["dataset"] = dataset_name
        frames.append(data)

    return pd.concat(frames, ignore_index=True)


def summarize_metrics(record_level: pd.DataFrame) -> pd.DataFrame:
    summary = (
        record_level.groupby(["dataset", "tokenizer_name"], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .sort_values(
            ["dataset", "tokenizer_name"],
            key=lambda values: values.map(
                {name: index for index, name in enumerate(DATASET_ORDER + TOKENIZER_ORDER)}
            ).fillna(len(DATASET_ORDER) + len(TOKENIZER_ORDER)),
        )
    )
    summary["estimated_cost_per_100_records"] = summary["estimated_cost"] * 100
    return summary


def calculate_winner_rates(record_level: pd.DataFrame) -> pd.DataFrame:
    token_counts = record_level.loc[
        record_level["test_name"] == "token_count",
        ["dataset", "record_id", "tokenizer_name", "token_count"],
    ].dropna(subset=["token_count"])
    winners = token_counts.loc[
        token_counts.groupby(["dataset", "record_id"])["token_count"].idxmin()
    ]
    rates = (
        winners.groupby(["dataset", "tokenizer_name"], as_index=False)
        .size()
        .rename(columns={"size": "winning_records"})
    )
    totals = winners.groupby("dataset", as_index=False).size().rename(columns={"size": "total_records"})
    rates = rates.merge(totals, on="dataset", how="left")
    rates["win_rate"] = rates["winning_records"] / rates["total_records"]

    complete_index = pd.MultiIndex.from_product(
        [DATASET_ORDER, TOKENIZER_ORDER], names=["dataset", "tokenizer_name"]
    )
    rates = (
        rates.set_index(["dataset", "tokenizer_name"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    rates["total_records"] = rates["dataset"].map(dict(zip(totals["dataset"], totals["total_records"])))
    rates["win_rate"] = rates["winning_records"] / rates["total_records"]
    return rates


def grouped_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    log_scale: bool = False,
) -> None:
    datasets = DATASET_ORDER
    width = 0.19
    x_positions = range(len(datasets))

    for offset_index, tokenizer in enumerate(TOKENIZER_ORDER):
        values = [
            summary.loc[
                (summary["dataset"] == dataset) & (summary["tokenizer_name"] == tokenizer),
                metric,
            ].iloc[0]
            for dataset in datasets
        ]
        offset = (offset_index - (len(TOKENIZER_ORDER) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=tokenizer,
            color=PALETTE[tokenizer],
        )

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    if log_scale:
        ax.set_yscale("log")


def plot_nsl_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    heatmap_data = (
        summary.pivot(index="dataset", columns="tokenizer_name", values="nsl")
        .reindex(index=DATASET_ORDER, columns=TOKENIZER_ORDER)
    )
    image = ax.imshow(heatmap_data, cmap="RdYlGn_r", vmin=0.9, vmax=1.12, aspect="auto")

    ax.set_title("Normalized sequence length vs GPT-2", fontweight="bold", pad=12)
    ax.set_xticks(range(len(TOKENIZER_ORDER)))
    ax.set_xticklabels(TOKENIZER_ORDER)
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels(DATASET_ORDER)
    ax.tick_params(axis="x", labelrotation=30)

    for row_index, dataset in enumerate(DATASET_ORDER):
        for column_index, tokenizer in enumerate(TOKENIZER_ORDER):
            value = heatmap_data.loc[dataset, tokenizer]
            ax.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="#1A1A1A")

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Relative token length")


def plot_winner_rates(ax: plt.Axes, winner_rates: pd.DataFrame) -> None:
    datasets = DATASET_ORDER
    width = 0.19
    x_positions = range(len(datasets))

    for offset_index, tokenizer in enumerate(TOKENIZER_ORDER):
        values = [
            winner_rates.loc[
                (winner_rates["dataset"] == dataset) & (winner_rates["tokenizer_name"] == tokenizer),
                "win_rate",
            ].iloc[0]
            for dataset in datasets
        ]
        offset = (offset_index - (len(TOKENIZER_ORDER) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=tokenizer,
            color=PALETTE[tokenizer],
        )

    ax.set_title("Most compact tokenizer by record", fontweight="bold")
    ax.set_ylabel("Share of records won")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)


def build_chart(summary: pd.DataFrame, winner_rates: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1],
        left=0.08,
        right=0.94,
        top=0.82,
        bottom=0.14,
        hspace=0.62,
        wspace=0.34,
    )
    heatmap_ax = figure.add_subplot(grid[0, :])
    pieces_ax = figure.add_subplot(grid[1, 0])
    winners_ax = figure.add_subplot(grid[1, 1])

    figure.suptitle("Tokenizer efficiency depends on text domain", fontsize=20, fontweight="bold", y=0.98)

    plot_nsl_heatmap(heatmap_ax, summary)
    grouped_bars(
        pieces_ax,
        summary,
        "pieces_per_word",
        "Average subword pieces per word",
        "Pieces per word",
    )
    plot_winner_rates(winners_ax, winner_rates)

    handles, labels = pieces_ax.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=len(TOKENIZER_ORDER),
        frameon=False,
    )
    for ax in [heatmap_ax, pieces_ax, winners_ax]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    figure.text(
        0.5,
        0.045,
        "Lower normalized sequence length and fewer pieces per word indicate more compact tokenization. Winner rate shows how often each tokenizer produced the fewest tokens for individual records.",
        ha="center",
        color="#444444",
    )
    figure.savefig(CHART_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    record_level = load_record_level_results()
    summary = summarize_metrics(record_level)
    winner_rates = calculate_winner_rates(record_level)
    SUMMARY_OUTPUT.parent.mkdir(exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    winner_rates.to_csv(WINNER_OUTPUT, index=False)
    build_chart(summary, winner_rates)

    print(f"Wrote summary data to {SUMMARY_OUTPUT}")
    print(f"Wrote winner rates to {WINNER_OUTPUT}")
    print(f"Wrote chart to {CHART_OUTPUT}")


if __name__ == "__main__":
    main()

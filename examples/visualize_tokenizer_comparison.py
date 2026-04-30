from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).with_name("results")
SUMMARY_OUTPUT = RESULTS_DIR / "combined_tokenizer_dataset_summary.csv"
WINNER_OUTPUT = RESULTS_DIR / "combined_tokenizer_winner_rates.csv"
CHART_OUTPUT = RESULTS_DIR / "combined_tokenizer_dataset_comparison.png"

DATASET_FILES = {
    "Wikipedia": "wikipedia_tokenizer_comparison_record_level_results.csv",
    "Common Crawl": "common_crawl_tokenizer_comparison_record_level_results.csv",
    "OpenSubtitles": "opensubtitles_tokenizer_comparison_record_level_results.csv",
}
DATASET_ORDER = list(DATASET_FILES)
TOKENIZER_ORDER = ["gpt2", "llama-sp", "t5", "wordpiece"]
COLORS = {
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


def main() -> None:
    record_level = load_results()
    summary = make_summary(record_level)
    winner_rates = make_winner_rates(record_level)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    winner_rates.to_csv(WINNER_OUTPUT, index=False)
    save_chart(summary, winner_rates)

    print(f"Wrote summary data to {SUMMARY_OUTPUT}")
    print(f"Wrote winner rates to {WINNER_OUTPUT}")
    print(f"Wrote chart to {CHART_OUTPUT}")


def load_results() -> pd.DataFrame:
    frames = []
    for dataset_name, file_name in DATASET_FILES.items():
        path = RESULTS_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")

        frame = pd.read_csv(path)
        frame["dataset"] = dataset_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def make_summary(record_level: pd.DataFrame) -> pd.DataFrame:
    summary = (
        record_level.groupby(["dataset", "tokenizer_name"], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .sort_values(["dataset", "tokenizer_name"], key=sort_key)
    )
    summary["estimated_cost_per_100_records"] = summary["estimated_cost"] * 100
    return summary


def make_winner_rates(record_level: pd.DataFrame) -> pd.DataFrame:
    token_counts = record_level.loc[
        record_level["test_name"] == "token_count",
        ["dataset", "record_id", "tokenizer_name", "token_count"],
    ].dropna(subset=["token_count"])

    winners = token_counts.loc[token_counts.groupby(["dataset", "record_id"])["token_count"].idxmin()]
    rates = (
        winners.groupby(["dataset", "tokenizer_name"], as_index=False)
        .size()
        .rename(columns={"size": "winning_records"})
    )
    totals = winners.groupby("dataset", as_index=False).size().rename(columns={"size": "total_records"})

    all_pairs = pd.MultiIndex.from_product(
        [DATASET_ORDER, TOKENIZER_ORDER],
        names=["dataset", "tokenizer_name"],
    )
    rates = rates.set_index(["dataset", "tokenizer_name"]).reindex(all_pairs, fill_value=0).reset_index()
    rates["total_records"] = rates["dataset"].map(dict(zip(totals["dataset"], totals["total_records"])))
    rates["win_rate"] = rates["winning_records"] / rates["total_records"]
    return rates


def sort_key(values: pd.Series) -> pd.Series:
    order = {name: index for index, name in enumerate(DATASET_ORDER + TOKENIZER_ORDER)}
    return values.map(order).fillna(len(order))


def save_chart(summary: pd.DataFrame, winner_rates: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.62, wspace=0.34)
    heatmap_ax = figure.add_subplot(grid[0, :])
    pieces_ax = figure.add_subplot(grid[1, 0])
    winners_ax = figure.add_subplot(grid[1, 1])

    figure.suptitle("Tokenizer efficiency depends on text domain", fontsize=20, fontweight="bold", y=0.98)
    plot_nsl_heatmap(heatmap_ax, summary)
    plot_grouped_bars(
        pieces_ax,
        summary,
        metric="pieces_per_word",
        title="Average subword pieces per word",
        ylabel="Pieces per word",
    )
    plot_grouped_bars(
        winners_ax,
        winner_rates,
        metric="win_rate",
        title="Most compact tokenizer by record",
        ylabel="Share of records won",
        percent_axis=True,
        y_limit=(0, 1),
    )

    handles, labels = pieces_ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=4, frameon=False)
    figure.text(
        0.5,
        0.045,
        "Lower normalized sequence length and fewer pieces per word indicate more compact tokenization. Winner rate shows how often each tokenizer produced the fewest tokens for individual records.",
        ha="center",
        color="#444444",
    )
    figure.savefig(CHART_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_grouped_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    percent_axis: bool = False,
    y_limit: tuple[float, float] | None = None,
) -> None:
    width = 0.19
    x_positions = list(range(len(DATASET_ORDER)))

    for index, tokenizer in enumerate(TOKENIZER_ORDER):
        values = []
        for dataset in DATASET_ORDER:
            value = frame.loc[
                (frame["dataset"] == dataset) & (frame["tokenizer_name"] == tokenizer),
                metric,
            ].iloc[0]
            values.append(value)

        offset = (index - (len(TOKENIZER_ORDER) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=tokenizer,
            color=COLORS[tokenizer],
        )

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DATASET_ORDER, rotation=15, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    if percent_axis:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def plot_nsl_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    heatmap = summary.pivot(index="dataset", columns="tokenizer_name", values="nsl")
    heatmap = heatmap.reindex(index=DATASET_ORDER, columns=TOKENIZER_ORDER)
    image = ax.imshow(heatmap, cmap="RdYlGn_r", vmin=0.9, vmax=1.12, aspect="auto")

    ax.set_title("Normalized sequence length vs GPT-2", fontweight="bold", pad=12)
    ax.set_xticks(range(len(TOKENIZER_ORDER)))
    ax.set_xticklabels(TOKENIZER_ORDER)
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels(DATASET_ORDER)
    ax.tick_params(axis="x", labelrotation=30)

    for row_index, dataset in enumerate(DATASET_ORDER):
        for column_index, tokenizer in enumerate(TOKENIZER_ORDER):
            ax.text(column_index, row_index, f"{heatmap.loc[dataset, tokenizer]:.3f}", ha="center", va="center")

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Relative token length")


if __name__ == "__main__":
    main()

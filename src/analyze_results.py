import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_latest_run() -> pd.DataFrame:
    records = []
    with open(RESULTS_DIR / "metrics.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    latest_run = df["run_id"].max()
    return df[df["run_id"] == latest_run].copy(), latest_run


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("variant")
        .agg(
            attack_success_rate=("compliance", "mean"),
            explanation_similarity_mean=("explanation_similarity", "mean"),
            explanation_similarity_std=("explanation_similarity", "std"),
            prompt_perplexity_mean=("prompt_perplexity", "mean"),
            prompt_perplexity_std=("prompt_perplexity", "std"),
        )
        .reset_index()
    )
    return grouped


def plot_bars(summary: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=summary, x="variant", y=metric, palette="Set2")
    plt.ylabel(ylabel)
    plt.xlabel("Prompt variant")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="prompt_perplexity",
        y="explanation_similarity",
        hue="variant",
        style="variant",
        palette="Set1",
    )
    plt.xlabel("Prompt perplexity (gpt2 proxy)")
    plt.ylabel("Explanation similarity")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close()


def main():
    df, run_id = load_latest_run()
    summary = summarize(df)
    summary.to_csv(RESULTS_DIR / "aggregated_metrics.csv", index=False)
    summary.to_json(RESULTS_DIR / "aggregated_metrics.json", orient="records", indent=2)

    plot_bars(summary, "attack_success_rate", "Attack success rate", "asr_bar.png")
    plot_bars(
        summary,
        "explanation_similarity_mean",
        "Explanation similarity (cosine)",
        "sim_bar.png",
    )
    plot_bars(
        summary,
        "prompt_perplexity_mean",
        "Prompt perplexity (gpt2)",
        "ppl_bar.png",
    )
    plot_scatter(df, "perplexity_vs_similarity.png")

    corr = df[["prompt_perplexity", "explanation_similarity"]].corr().iloc[0, 1]
    stats = {
        "run_id": run_id,
        "n_samples": len(df),
        "perplexity_similarity_correlation": corr,
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

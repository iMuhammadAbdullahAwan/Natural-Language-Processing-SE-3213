"""
Visualization utilities for evaluation metrics and analyses.
Saves plots as PNG images for inclusion in reports.
"""
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


MAIN_METRICS = [
    ("word_accuracy", "Word Accuracy"),
    ("character_accuracy", "Character Accuracy"),
    ("sentence_accuracy", "Sentence Accuracy"),
    ("bleu_score", "BLEU Score"),
    ("rouge_l", "ROUGE-L"),
    ("meteor_score", "METEOR"),
]


def _ensure_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_metric_bars(results: Dict[str, Dict], output_dir: Path) -> List[Path]:
    """Create grouped bar charts for main metrics across models.

    Args:
        results: evaluator.results dict where each model has a 'metrics' mapping.
        output_dir: directory to save images.

    Returns:
        List of saved image paths.
    """
    _ensure_dir(output_dir)
    image_paths: List[Path] = []

    # Build DataFrame of metrics x models
    rows = []
    for model_key, model_res in results.items():
        metrics = model_res.get("metrics", {})
        row = {"Model": model_res.get("model_type", model_key)}
        for m_key, _m_name in MAIN_METRICS:
            row[m_key] = float(metrics.get(m_key, 0.0))
        # Edit distance separately
        row["avg_edit_distance"] = float(metrics.get("avg_edit_distance", 0.0))
        rows.append(row)

    if not rows:
        return image_paths

    df = pd.DataFrame(rows)

    # One combined bar chart (normalized metrics 0..1; edit distance on separate plot)
    metric_cols = [k for k, _ in MAIN_METRICS]
    pretty_names = {k: v for k, v in MAIN_METRICS}
    df_pretty = df.rename(columns=pretty_names)

    # Combined bar chart
    plt.figure(figsize=(12, 6))
    melted = df_pretty.melt(id_vars=["Model"], value_vars=list(pretty_names.values()),
                            var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.title("Model Performance Across Metrics")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    out1 = output_dir / "metrics_overview_bar.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    image_paths.append(out1)

    # Edit distance bar (lower is better)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="avg_edit_distance", palette="viridis")
    plt.title("Average Edit Distance (Lower is Better)")
    plt.xlabel("Model")
    plt.ylabel("Avg Edit Distance")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out2 = output_dir / "avg_edit_distance_bar.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    image_paths.append(out2)

    return image_paths


def plot_error_analysis(results: Dict[str, Dict], output_dir: Path) -> List[Path]:
    """Plot stacked/grouped bars for error types per model."""
    _ensure_dir(output_dir)
    rows = []
    for model_key, model_res in results.items():
        ea = model_res.get("error_analysis", {})
        if not ea:
            continue
        row = {
            "Model": model_res.get("model_type", model_key),
            "Substitution": ea.get("substitution_errors", 0),
            "Insertion": ea.get("insertion_errors", 0),
            "Deletion": ea.get("deletion_errors", 0),
            "Word Order": ea.get("word_order_errors", 0),
        }
        rows.append(row)

    image_paths: List[Path] = []
    if not rows:
        return image_paths

    df = pd.DataFrame(rows)
    melted = df.melt(id_vars=["Model"], var_name="Error Type", value_name="Count")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Error Type", y="Count", hue="Model")
    plt.title("Error Type Distribution by Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = output_dir / "error_types_by_model.png"
    plt.savefig(out, dpi=200)
    plt.close()
    image_paths.append(out)
    return image_paths


def plot_length_analysis(results: Dict[str, Dict], output_dir: Path) -> List[Path]:
    """Plot predicted vs reference average lengths per model and length ratio boxplot."""
    _ensure_dir(output_dir)
    rows = []
    for model_key, model_res in results.items():
        la = model_res.get("length_analysis", {})
        if not la:
            continue
        rows.append({
            "Model": model_res.get("model_type", model_key),
            "Avg Pred Length": float(la.get("avg_pred_length", 0.0)),
            "Avg Ref Length": float(la.get("avg_ref_length", 0.0)),
            "Length Ratio": float(la.get("length_ratio", 0.0)),
        })

    image_paths: List[Path] = []
    if not rows:
        return image_paths

    df = pd.DataFrame(rows)

    # Side-by-side bars for avg lengths
    plt.figure(figsize=(10, 6))
    df_m = df.melt(id_vars=["Model"], value_vars=["Avg Pred Length", "Avg Ref Length"],
                   var_name="Type", value_name="Length")
    sns.barplot(data=df_m, x="Type", y="Length", hue="Model")
    plt.title("Average Sentence Lengths (Words)")
    plt.tight_layout()
    out1 = output_dir / "avg_lengths_by_model.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    image_paths.append(out1)

    # Length ratio bars
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="Length Ratio", palette="mako")
    plt.axhline(1.0, linestyle="--", color="red", alpha=0.6)
    plt.title("Length Ratio (Pred/Ref) by Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out2 = output_dir / "length_ratio_by_model.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    image_paths.append(out2)

    return image_paths


def save_all_plots(results: Dict[str, Dict], comparison: Dict, images_dir: str | Path) -> List[Path]:
    """High-level function to save all standard plots.

    Args:
        results: evaluator.results with per-model blocks including metrics and analyses.
        comparison: output from evaluator.compare_models().
        images_dir: directory path to save the images.

    Returns:
        List of all saved image file paths.
    """
    out_dir = Path(images_dir)
    _ensure_dir(out_dir)

    saved: List[Path] = []
    saved += plot_metric_bars(results, out_dir)
    saved += plot_error_analysis(results, out_dir)
    saved += plot_length_analysis(results, out_dir)

    # If comparison has metric_summary DataFrame, save it as an image-like table using matplotlib
    metric_summary = comparison.get("metric_summary")
    if metric_summary is not None and isinstance(metric_summary, pd.DataFrame):
        fig, ax = plt.subplots(figsize=(max(8, metric_summary.shape[1] * 1.6),
                                        max(3, metric_summary.shape[0] * 0.6)))
        ax.axis('off')
        tbl = ax.table(cellText=metric_summary.round(4).values,
                       colLabels=list(metric_summary.columns),
                       loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        out = out_dir / "metric_summary_table.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)

    return saved

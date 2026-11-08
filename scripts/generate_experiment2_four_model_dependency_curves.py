"""Plot Experiment 2 dependency-correctness curves for four model backbones.

The script reads the completed Experiment 2 summary files and creates a 2 x 2
panel figure. Only dependency correctness is plotted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"

MODEL_INPUTS = [
    (
        "DeepSeek-V4-Flash",
        METRICS_DIR
        / "experiment2_llm_deepseek_summary_toolsandbox_original_100_run.json",
    ),
    (
        "GPT-OSS-120B",
        METRICS_DIR
        / "experiment2_llm_siliconflow_summary_gpt_oss_120b_toolsandbox_original_100.json",
    ),
    (
        "Gemma-4-31B-it",
        METRICS_DIR
        / "experiment2_llm_siliconflow_summary_gemma_4_31b_it_toolsandbox_original_100.json",
    ),
    (
        "Qwen3.6-35B-A3B",
        METRICS_DIR
        / "experiment2_llm_siliconflow_summary_qwen3_6_35b_a3b_toolsandbox_original_100.json",
    ),
]

LEVELS = [
    "L1_SINGLE_TOOL",
    "L2_EXPLICIT_DEP",
    "L3_IMPLICIT_SIDE_EFFECT",
]
LEVEL_LABELS = [
    "L1\nSingle tool",
    "L2\nExplicit dep.",
    "L3\nImplicit dep.",
]
STRATEGIES = ["direct", "cot", "react", "self_refine"]

# Manually audited corrections, stored as (correct cases, total cases). Keeping
# these overrides here preserves the original run summaries.
CORRECTED_COUNTS = {
    (
        "experiment2_llm_deepseek_summary_toolsandbox_original_100_run.json",
        "react",
        "L2_EXPLICIT_DEP",
    ): (19, 24),
    (
        "experiment2_llm_deepseek_summary_toolsandbox_original_100_run.json",
        "react",
        "L3_IMPLICIT_SIDE_EFFECT",
    ): (42, 68),
    (
        "experiment2_llm_siliconflow_summary_gemma_4_31b_it_toolsandbox_original_100.json",
        "cot",
        "L2_EXPLICIT_DEP",
    ): (16, 24),
}

STYLE = {
    "direct": {"label": "Direct", "color": "#E88787", "marker": "o"},
    "cot": {"label": "CoT", "color": "#F59E0B", "marker": "s"},
    "react": {"label": "ReAct", "color": "#14B8A6", "marker": "^"},
    "self_refine": {
        "label": "Self-Refine",
        "color": "#7C3AED",
        "marker": "D",
    },
}

INK = "#111827"
AXIS_GRAY = "#334155"
GRID_GRAY = "#CBD5E1"


def configure_style() -> None:
    """Set publication-oriented Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS_GRAY,
            "axes.linewidth": 0.85,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_dependency_curves(path: Path) -> dict[str, list[float]]:
    """Load dependency-correctness percentages for one model."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Experiment 2 summary: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        (row["strategy"], row["difficulty_level"]): row
        for row in payload["group_metrics"]
        if row.get("source") == "all"
    }

    missing = [
        (strategy, level)
        for strategy in STRATEGIES
        for level in LEVELS
        if (strategy, level) not in rows
    ]
    if missing:
        raise ValueError(f"Missing strategy/level rows in {path.name}: {missing}")

    curves: dict[str, list[float]] = {}
    for strategy in STRATEGIES:
        curves[strategy] = []
        for level in LEVELS:
            correction = CORRECTED_COUNTS.get((path.name, strategy, level))
            if correction is None:
                rate = rows[(strategy, level)]["dependency_correctness_rate"]
            else:
                correct_cases, total_cases = correction
                rate = correct_cases / total_cases
            curves[strategy].append(100.0 * rate)
    return curves


def polish_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.12, 2.12)
    ax.set_ylim(-3, 103)
    ax.set_xticks(np.arange(3), LEVEL_LABELS)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.grid(axis="y", color=GRID_GRAY, linewidth=0.60, alpha=0.42)
    ax.tick_params(axis="both", length=0, colors=AXIS_GRAY)
    ax.spines["left"].set_color(AXIS_GRAY)
    ax.spines["bottom"].set_color(AXIS_GRAY)
    ax.set_axisbelow(True)


def plot_panel(
    ax: plt.Axes,
    curves: dict[str, list[float]],
    panel_label: str,
    model_name: str,
) -> None:
    """Draw one model panel in the style of the original dependency curve."""
    x = np.arange(3)

    # Layered translucent bands preserve the visual language of the source plot.
    bands = [
        (np.zeros_like(x, dtype=float), curves["direct"], "direct", 0.15),
        (curves["direct"], curves["cot"], "cot", 0.22),
        (curves["cot"], curves["react"], "react", 0.20),
        (curves["react"], curves["self_refine"], "self_refine", 0.18),
    ]
    for lower, upper, strategy, alpha in bands:
        ax.fill_between(
            x,
            lower,
            upper,
            color=STYLE[strategy]["color"],
            alpha=alpha,
            linewidth=0,
            interpolate=True,
            zorder=1,
        )

    for strategy in STRATEGIES:
        style = STYLE[strategy]
        is_baseline = strategy in {"direct", "cot"}
        use_small_marker = strategy in {"direct", "self_refine"}
        ax.plot(
            x,
            curves[strategy],
            color=style["color"],
            linewidth=1.95 if strategy == "self_refine" else 1.75,
            linestyle="-",
            alpha=0.56 if is_baseline else 0.78,
            marker=style["marker"],
            markersize=2.3 if use_small_marker else 2.65,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.48,
            label=style["label"],
            zorder=5,
        )
        ax.scatter(
            x,
            curves[strategy],
            s=6.75 if use_small_marker else 8.75,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.48,
            alpha=0.88 if is_baseline else 0.98,
            zorder=6,
        )

    polish_axis(ax)
    ax.set_title(
        f"{panel_label}) {model_name} dependency curve",
        loc="left",
        fontweight="bold",
        color=INK,
        pad=7,
    )


def main() -> None:
    configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.5),
        dpi=300,
        sharex=False,
        sharey=True,
    )

    for panel_label, ax, (model_name, input_path) in zip(
        "abcd", axes.flat, MODEL_INPUTS, strict=True
    ):
        plot_panel(ax, load_dependency_curves(input_path), panel_label, model_name)

    fig.supylabel("Dependency correctness (%)", x=0.02, fontsize=9.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=True,
        borderpad=0.55,
        handlelength=2.7,
        columnspacing=1.8,
    )
    legend.get_frame().set_edgecolor("#E5E7EB")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.97)

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.16,
        top=0.955,
        wspace=0.12,
        hspace=0.34,
    )

    output_stem = FIGURES_DIR / "experiment2_four_model_dependency_curves"
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    for suffix in (".png", ".pdf", ".svg"):
        print(f"Saved: {output_stem.with_suffix(suffix)}")


if __name__ == "__main__":
    main()

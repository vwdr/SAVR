#!/usr/bin/env python3
"""Build the five standalone figures for the SAVR MIT URTC poster."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "poster" / "mit_urtc_savr" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Neutral-first palette. Gold identifies temporal reuse; red is reserved for one
# high-consequence failure callout. The poster itself already supplies red structure.
INK = "#171717"
SLATE = "#5F6872"
MID = "#A7ADB4"
PALE = "#ECEFF1"
GOLD = "#C99524"
GOLD_PALE = "#F3E5BD"
RED = "#8B1E2D"
WHITE = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }
)


def clean_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=PALE, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def rounded_box(
    ax, xy, width, height, title, subtitle="", fc=WHITE, ec=INK, lw=2.0,
    title_size=13, subtitle_size=10
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontsize=title_size, weight="bold")
    if subtitle:
        ax.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center", fontsize=subtitle_size, color=SLATE)
    return patch


def arrow(ax, start, end, color=INK, connectionstyle="arc3", lw=2.2, zorder=3):
    p = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def build_methods():
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(0.1, 5.18, "HOW TEMPORAL VISUAL REUSE WAS EVALUATED", fontsize=23, weight="bold", va="top")

    rounded_box(ax, (0.20, 2.15), 2.05, 1.45, "CURRENT\nINPUTS", "scene + wrist images\ninstruction + robot state", fc=PALE, title_size=13, subtitle_size=10)
    rounded_box(ax, (2.75, 2.15), 1.85, 1.45, "OpenVLA-OFT", "visual encoder\n+ policy", title_size=14, subtitle_size=10.5)
    arrow(ax, (2.25, 2.88), (2.75, 2.88))

    rows = [
        (3.65, "COMPLETE PREFIX", "reuse scene + wrist representations"),
        (2.20, "SCENE CAMERA", "reuse scene and refresh wrist"),
        (0.75, "FINE-GRAINED K/V", "reuse selected layers + image regions"),
    ]
    for idx, (y, title, rule) in enumerate(rows):
        rounded_box(
            ax, (5.25, y), 3.25, 1.02, title, rule,
            fc=GOLD_PALE if idx == 1 else WHITE, ec=GOLD,
            title_size=13.5, subtitle_size=10.3,
        )
        arrow(ax, (4.60, 2.88), (5.25, y + 0.51), color=GOLD,
              connectionstyle=f"arc3,rad={0.15 * (1-idx):.2f}", lw=1.9)

    ax.plot([8.92], [2.88], marker="o", markersize=7, color=INK, clip_on=False)
    for y, *_ in rows:
        arrow(ax, (8.50, y + 0.51), (8.92, 2.88), connectionstyle="arc3,rad=0.08", lw=1.7)

    rounded_box(ax, (9.25, 2.15), 1.85, 1.45, "ACTION\nCHUNK", "robot control output", fc=PALE, title_size=13, subtitle_size=10)
    arrow(ax, (8.92, 2.88), (9.25, 2.88))
    rounded_box(ax, (11.60, 2.15), 1.85, 1.45, "LIBERO", "closed-loop task", title_size=14, subtitle_size=10.5)
    arrow(ax, (11.10, 2.88), (11.60, 2.88))

    arrow(ax, (12.52, 2.15), (1.20, 2.15), color=SLATE, connectionstyle="arc3,rad=-0.23", lw=1.8, zorder=1)
    ax.text(7.05, 0.16, "Actions determine the next observation, so every strategy was evaluated in closed loop.",
            fontsize=14.2, weight="bold", ha="center", va="bottom")
    ax.text(11.75, 1.05, "next observation", fontsize=11.5, color=SLATE, ha="center")
    save(fig, "figure_1_methods_overview")


def build_complete_prefix():
    permissive_x = np.array([34.69, 43.29, 47.28, 56.62, 63.26, 65.21, 64.14, 75.00, 83.68])
    permissive_y = np.array([52, 21, 23, 12, 4, 0, 3, 1, 0])
    conservative_x = np.array([6.72, 10.57])
    conservative_y = np.array([96.67, 90.00])
    final_x, final_y = 0.9534, 98.57
    assert len(permissive_x) == 9 and np.isclose(final_y, 69 / 70 * 100, atol=0.01)

    fig = plt.figure(figsize=(18.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.15], wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    clean_axes(ax, "both")
    fig.suptitle("COMPLETE VISUAL-PREFIX REUSE", fontsize=23, weight="bold", y=0.98)
    fig.text(
        0.50,
        0.915,
        "1,160 primary closed-loop episodes | separate 50-episode timing pilot",
        ha="center",
        fontsize=13.5,
        color=SLATE,
    )
    ax.set_title("A  Closed-loop success versus reuse", loc="left", pad=12)
    ax.axhspan(98, 100, color=PALE, zorder=0)
    ax.axhline(98, color=MID, linestyle=(0, (5, 4)), linewidth=1.7)
    ax.text(88.5, 99.0, "within 2 percentage points of dense", ha="right", va="center", fontsize=12.5, color=SLATE)
    ax.scatter(permissive_x, permissive_y, s=105, color=SLATE, edgecolor=WHITE, linewidth=1.0,
               label="permissive settings (100 episodes each)", zorder=3)
    ax.scatter(conservative_x, conservative_y, s=125, marker="s", color=INK, edgecolor=WHITE, linewidth=1.0,
               label="conservative settings (30 episodes each)", zorder=4)
    ax.scatter([0], [100], s=190, marker="D", facecolor=WHITE, edgecolor=INK, linewidth=2.2,
               label="dense reference (100/100)", zorder=5)
    ax.scatter([final_x], [final_y], s=230, marker="*", facecolor=GOLD, edgecolor=INK, linewidth=1.1,
               label="final conservative setting (69/70)", zorder=6)
    ax.annotate(
        "69/70 success\nonly 0.95% reuse",
        (final_x, final_y),
        xytext=(19, 84),
        textcoords="data",
        fontsize=14,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.7),
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GOLD, lw=1.6),
    )
    ax.annotate(
        "best permissive setting\n52/100 success at 34.69% reuse",
        (34.69, 52),
        xytext=(50, 64),
        fontsize=13.5,
        arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.30", fc=WHITE, ec=MID),
    )
    ax.set_xlim(-2, 90)
    ax.set_ylim(-3, 103)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("policy queries reusing the complete visual prefix (%)")
    ax.set_ylabel("terminal task success (%)")
    handles, labels = ax.get_legend_handles_labels()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("B  Measured CUDA-time budget", loc="left", pad=12)
    ax2.barh([0], [84.126], color=PALE, edgecolor=INK, height=0.50, label="remaining policy computation")
    ax2.barh([0], [15.874], left=[84.126], color=GOLD, edgecolor=INK, height=0.50, label="visual encoder + projector")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.70, 0.72)
    ax2.set_yticks([])
    ax2.set_xlabel("share of measured query CUDA time (%)")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.grid(axis="x", color=PALE, linewidth=1.2)
    ax2.text(42.0, 0, "84.1%\nother computation", ha="center", va="center", fontsize=14, weight="bold")
    ax2.text(92.0, 0, "15.9%\nvisual", ha="center", va="center", fontsize=13, weight="bold")
    ax2.text(50, -0.50, "50-episode timing pilot | 662 steady queries", ha="center", fontsize=12.5, color=SLATE)
    ax2.text(50, 0.56, "Even zero-cost visual processing\nat most 1.189x theoretical speedup",
             ha="center", va="center", fontsize=15.2, weight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GOLD, lw=1.6))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.37, 0.02), frameon=False,
               fontsize=11.8, ncol=2, handletextpad=0.5, columnspacing=1.4)
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.06, right=0.99)
    save(fig, "figure_2_complete_prefix_results")


def build_scene_camera():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.2), gridspec_kw={"wspace": 0.38})
    fig.suptitle("SCENE-CAMERA REUSE", fontsize=25, weight="bold", y=0.97)
    fig.text(0.50, 0.905, "Same observed success count, but no end-to-end speedup", ha="center", fontsize=17, color=SLATE)

    ax = axes[0]
    clean_axes(ax, "y")
    labels = ["Batched dense", "Scene-camera reuse"]
    success = [67, 67]
    colors = [INK, GOLD]
    bars = ax.bar(labels, success, color=colors, width=0.60, zorder=3)
    ax.set_ylim(0, 72)
    ax.set_ylabel("successful episodes (out of 70)")
    ax.set_title("A  Closed-loop success", loc="left", fontsize=20, pad=10)
    for bar, value in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value}/70", ha="center", fontsize=20, weight="bold")
    ax.text(0.5, 6, "same observed success count", ha="center", fontsize=14.5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=MID))
    ax.tick_params(axis="x", labelrotation=0)

    ax = axes[1]
    ax.set_title("B  Complete-query wall time", loc="left", fontsize=20, pad=10)
    times = [1188.28, 1190.97]
    y = [1, 0]
    ax.hlines(y, xmin=1185, xmax=1194, color=PALE, linewidth=8, zorder=0)
    ax.scatter(times, y, s=230, color=colors, edgecolor=WHITE, linewidth=1.5, zorder=3)
    for val, yy in zip(times, y):
        ax.text(val, yy + 0.17, f"{val:,.2f} ms", ha="center", fontsize=18, weight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(1185, 1194)
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("wall time per query (ms)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=PALE, linewidth=1.2)
    ax.tick_params(axis="y", length=0)
    ax.text(1189.63, -0.45, "+2.69 ms (+0.23%) with reuse", ha="center", fontsize=14.5, weight="bold",
            color=RED, bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=RED, lw=1.3))

    fig.text(0.50, 0.045, "Scene representation reused on 25.24% of queries | 70 episodes per condition from paired initial states",
             ha="center", fontsize=14.5, weight="bold")
    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.10, right=0.98)
    save(fig, "figure_3_scene_camera_results")


def build_kv():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.2), gridspec_kw={"width_ratios": [0.82, 1.35], "wspace": 0.36})
    fig.suptitle("FINE-GRAINED K/V REUSE", fontsize=25, weight="bold", y=0.97)
    fig.text(0.50, 0.905, "Lower query time did not preserve closed-loop task success", ha="center", fontsize=17, color=SLATE)

    ax = axes[0]
    clean_axes(ax, "y")
    vals = [100.0, 77.4031]
    bars = ax.bar(["Dense", "K/V reuse"], vals, color=[INK, GOLD], width=0.62, zorder=3)
    ax.set_ylim(0, 112)
    ax.set_ylabel("median complete-query time\n(normalized, dense = 100%)")
    ax.set_title("A  Controlled systems test", loc="left", fontsize=20, pad=12)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.2, f"{val:.1f}%", ha="center", fontsize=18, weight="bold")
    ax.text(0.50, 29, "22.60% lower\nmedian query time", ha="center", fontsize=16, weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=GOLD, lw=1.6))
    ax.text(0.5, 4, "97 policy calls", ha="center", fontsize=13, color=SLATE)

    ax = axes[1]
    clean_axes(ax, "y")
    suites = ["Spatial", "Object", "Goal", "LIBERO-10", "Overall"]
    dense = np.array([46.67, 90.00, 43.33, 46.67, 56.67])
    x = np.arange(len(suites))
    width = 0.34
    b1 = ax.bar(x - width / 2, dense, width, color=INK, label="Dense inference", zorder=3)
    ax.scatter(x + width / 2, np.full_like(x, 1.2, dtype=float), marker="x", s=110, linewidth=2.7, color=RED, zorder=4)
    for i, (bar, val) in enumerate(zip(b1, dense)):
        label = f"{val:.1f}%" if i < 4 else f"{val:.1f}%\n68/120"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.1, label, ha="center", fontsize=14.5, weight="bold")
    ax.set_xticks(x, suites, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("terminal task success (%)")
    ax.set_title("B  Paired closed-loop evaluation", loc="left", fontsize=20, pad=12)
    for i in range(4):
        ax.text(i + width / 2, 4.0, "0/30", ha="center", fontsize=10.5, weight="bold", color=RED)
    ax.text(4 + width / 2, 4.0, "0/120", ha="center", fontsize=10.5, weight="bold", color=RED)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=INK, label="Dense inference"),
            Line2D([0], [0], marker="x", color="none", markeredgecolor=RED, markeredgewidth=2.4,
                   markersize=9, label="Fine-grained K/V reuse"),
        ],
        frameon=False, loc="upper right", fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=13.5)
    fig.text(0.50, 0.075, "Closed-loop evaluation with 30 episodes per suite and condition (120 overall)",
             ha="center", fontsize=11.8, color=SLATE)
    fig.text(0.50, 0.035, "Timing and task success were measured in separate, explicitly labeled tests.",
             ha="center", fontsize=13.2, weight="bold")
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.10, right=0.98)
    save(fig, "figure_4_fine_grained_kv_results")


def build_feedback():
    fig, ax = plt.subplots(figsize=(14.5, 4.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.10, 4.42, "WHY OFFLINE SIMILARITY DOES NOT GUARANTEE CLOSED-LOOP SUCCESS", fontsize=20, weight="bold", va="top")
    ax.text(7.25, 3.78, "Offline replay follows the dense trajectory and omits this feedback loop.",
            ha="center", va="center", fontsize=13.0, color=SLATE)

    rounded_box(ax, (0.20, 2.35), 2.35, 0.92, "CURRENT\nOBSERVATION", "camera images + robot state", fc=PALE, title_size=10.8, subtitle_size=8.7)
    rounded_box(ax, (0.20, 1.10), 2.35, 0.92, "REUSED STATE", "representation from\nan earlier query", fc=GOLD_PALE, ec=GOLD, title_size=11.2, subtitle_size=8.5)
    rounded_box(ax, (3.35, 1.72), 2.30, 1.05, "MIXED-AGE STATE", "current + stale information", title_size=11.5, subtitle_size=9)
    rounded_box(ax, (6.45, 1.72), 2.30, 1.05, "ACTION CHANGES", "different action chunk", title_size=11.5, subtitle_size=9)
    rounded_box(ax, (9.55, 1.72), 2.30, 1.05, "NEXT\nOBSERVATION", "the robot follows\na new trajectory", fc=PALE, title_size=10.8, subtitle_size=8.4)
    rounded_box(ax, (12.45, 1.72), 1.82, 1.05, "NEXT REUSE", "decision uses the\nnew trajectory", title_size=11.0, subtitle_size=8.6)

    arrow(ax, (2.55, 2.81), (3.35, 2.34), color=INK)
    arrow(ax, (2.55, 1.56), (3.35, 2.15), color=GOLD)
    arrow(ax, (5.65, 2.24), (6.45, 2.24), color=INK)
    arrow(ax, (8.75, 2.24), (9.55, 2.24), color=INK)
    arrow(ax, (11.85, 2.24), (12.45, 2.24), color=INK)
    arrow(ax, (13.35, 1.72), (1.38, 1.10), color=SLATE, connectionstyle="arc3,rad=-0.22", lw=1.9)
    ax.text(7.40, 0.22, "A changed action shifts the observations on which every later cache decision is made.",
            ha="center", va="center", fontsize=13.2, weight="bold")
    save(fig, "figure_5_closed_loop_feedback")


def main():
    build_methods()
    build_complete_prefix()
    build_scene_camera()
    build_kv()
    build_feedback()
    print(f"Wrote five PDF figures and five PNG previews to {OUT}")


if __name__ == "__main__":
    main()

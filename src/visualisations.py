import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json

# Load the per-prompt and per-category results from the previous step
prompt_df   = pd.read_csv("results/per_prompt_comparison.csv")
category_df = pd.read_csv("results/per_category_comparison.csv")

categories = ["unanswerable", "open_ended", "factual"]
cat_labels = ["Unanswerable", "Open Ended", "Factual"]

colors = {
    "normal":    "#2563EB",
    "misrouted": "#F59E0B",
    "top1":      "#DC2626",
}

row_labels = [
    "Normal (mean ± std)",
    "Random (mean ± std)",
    "Top-1 (mean ± std)",
    "Diff: Random − Normal",
    "Diff: Top-1 − Normal",
]

condition_order = [("normal", "Normal"), ("misrouted", "Random"), ("top1", "Top-1")]


def style_table(table, n_cols, n_data_rows):
    # Apply the dark navy header, the soft-pink rows for the difference lines at the bottom, and the pale-blue first column for row labels
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    for j in range(n_cols):
        table[0, j].set_facecolor("#1e3a5f")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(n_cols):
        table[n_data_rows - 1, j].set_facecolor("#fff7f7")
        table[n_data_rows - 2, j].set_facecolor("#fff7f7")
    for i in range(1, n_data_rows + 1):
        table[i, -1].set_facecolor("#f0f4ff")
        table[i, -1].set_text_props(fontweight="bold")


def truncate(text, n=48):
    # Cut long strings down to length n and stick an ellipsis on the end
    return text[:n] + "..." if len(text) > n else text


# Build Table 1: mean entropy per category for each condition, with the difference rows underneath
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
fig.suptitle("Table 1: Mean Semantic Entropy by Category and Routing Condition",
             fontsize=12, fontweight="bold", y=0.98)

# One row per condition showing mean ± std for each category
table_data = []
for condition in ["normal", "misrouted", "top1"]:
    row = []
    for cat in categories:
        r    = category_df[category_df["category"] == cat].iloc[0]
        mean = r[f"{condition}_mean_entropy"]
        std  = r[f"{condition}_std_entropy"]
        row.append(f"{mean:.4f} ± {std:.4f}")
    table_data.append(row)

# Two more rows showing how far Random and Top-1 shifted from the Normal baseline
for diff_pair in [("misrouted", "normal"), ("top1", "normal")]:
    row = []
    for cat in categories:
        r    = category_df[category_df["category"] == cat].iloc[0]
        diff = r[f"{diff_pair[0]}_mean_entropy"] - r[f"{diff_pair[1]}_mean_entropy"]
        sign = "+" if diff > 0 else ""
        row.append(f"{sign}{diff:.4f}")
    table_data.append(row)

table = ax.table(cellText=table_data, rowLabels=row_labels,
                 colLabels=cat_labels, cellLoc="center", loc="center")
style_table(table, len(cat_labels), len(row_labels))
plt.tight_layout()
plt.savefig("results/table_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/table_entropy.png")
plt.close()


# Build Table 2: same layout as Table 1, but using cluster counts instead of entropy
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
fig.suptitle("Table 2: Mean Cluster Count by Category and Routing Condition",
             fontsize=12, fontweight="bold", y=0.98)

table_data2 = []
for condition in ["normal", "misrouted", "top1"]:
    row = []
    for cat in categories:
        vals = prompt_df[prompt_df["category"] == cat][f"{condition}_clusters"]
        row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")
    table_data2.append(row)

for diff_pair in [("misrouted", "normal"), ("top1", "normal")]:
    row = []
    for cat in categories:
        subset = prompt_df[prompt_df["category"] == cat]
        diff   = subset[f"{diff_pair[0]}_clusters"].mean() - subset[f"{diff_pair[1]}_clusters"].mean()
        sign   = "+" if diff > 0 else ""
        row.append(f"{sign}{diff:.2f}")
    table_data2.append(row)

table2 = ax.table(cellText=table_data2, rowLabels=row_labels,
                  colLabels=cat_labels, cellLoc="center", loc="center")
style_table(table2, len(cat_labels), len(row_labels))
plt.tight_layout()
plt.savefig("results/table_clusters.png", dpi=150, bbox_inches="tight")
print("Saved: results/table_clusters.png")
plt.close()


# Bigger fonts for the main report figures, no error bars on top of the bars
TITLE_SIZE  = 20
LABEL_SIZE  = 18
TICK_SIZE   = 16
LEGEND_SIZE = 16

# Bar chart of mean entropy per category, with all three conditions side by side
fig, ax = plt.subplots(figsize=(14, 8))
x     = np.arange(len(categories))
width = 0.25

for i, (condition, label) in enumerate(condition_order):
    means  = [category_df[category_df["category"] == cat].iloc[0][f"{condition}_mean_entropy"]
              for cat in categories]
    # Shift each condition's bars left or right of centre so the three groups sit side by side rather than overlapping
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, label=label,
           color=colors[condition], alpha=0.85)

ax.set_title("Mean Semantic Entropy by Category: All Routing Conditions",
             fontsize=TITLE_SIZE, fontweight="bold", pad=15)
ax.set_ylabel("Semantic Entropy (nats)", fontsize=LABEL_SIZE)
ax.set_xlabel("Prompt Category", fontsize=LABEL_SIZE)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=TICK_SIZE)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/bar_chart_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/bar_chart_entropy.png")
plt.close()


# Bar chart of mean cluster count per category, same layout as the entropy one above
fig, ax = plt.subplots(figsize=(14, 8))

for i, (condition, label) in enumerate(condition_order):
    means  = [prompt_df[prompt_df["category"] == cat][f"{condition}_clusters"].mean()
              for cat in categories]
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, label=label,
           color=colors[condition], alpha=0.85)

ax.set_title("Mean Cluster Count by Category: All Routing Conditions",
             fontsize=TITLE_SIZE, fontweight="bold", pad=15)
ax.set_ylabel("Mean Number of Clusters", fontsize=LABEL_SIZE)
ax.set_xlabel("Prompt Category", fontsize=LABEL_SIZE)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=TICK_SIZE)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/bar_chart_clusters.png", dpi=150, bbox_inches="tight")
print("Saved: results/bar_chart_clusters.png")
plt.close()


# One box plot per category, with all three conditions sitting alongside each other
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
fig.suptitle("Distribution of Semantic Entropy by Category and Routing Condition",
             fontsize=TITLE_SIZE, fontweight="bold")

for ax, cat, cat_label in zip(axes, categories, cat_labels):
    subset = prompt_df[prompt_df["category"] == cat]
    data   = [subset["normal_entropy"].values,
              subset["misrouted_entropy"].values,
              subset["top1_entropy"].values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2.5))
    # Colour each box to match its routing condition
    for patch, (cond, _) in zip(bp["boxes"], condition_order):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.8)
    ax.set_title(cat_label, fontsize=TITLE_SIZE - 2, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Normal", "Random", "Top-1"], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylabel("Semantic Entropy (nats)" if ax == axes[0] else "",
                  fontsize=LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/box_plots_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/box_plots_entropy.png")
plt.close()


# Scatter plot of Normal entropy against the other two conditions, one point per prompt
fig, ax = plt.subplots(figsize=(9, 9))

# A separate marker style and colour for each condition + category combination, so all six groups can sit on the same plot without blending together
scatter_styles = {
    ("misrouted", "unanswerable"): {"color": "#2563EB", "marker": "o", "label": "Random – Unanswerable"},
    ("misrouted", "open_ended"):   {"color": "#16A34A", "marker": "o", "label": "Random – Open Ended"},
    ("misrouted", "factual"):      {"color": "#9333EA", "marker": "o", "label": "Random – Factual"},
    ("top1",      "unanswerable"): {"color": "#DC2626", "marker": "^", "label": "Top-1 – Unanswerable"},
    ("top1",      "open_ended"):   {"color": "#F59E0B", "marker": "^", "label": "Top-1 – Open Ended"},
    ("top1",      "factual"):      {"color": "#0891B2", "marker": "^", "label": "Top-1 – Factual"},
}

for (condition, cat), style in scatter_styles.items():
    subset = prompt_df[prompt_df["category"] == cat]
    ax.scatter(subset["normal_entropy"], subset[f"{condition}_entropy"],
               color=style["color"], marker=style["marker"],
               alpha=0.75, s=100, label=style["label"])

# Set both axes to the same range and draw a y = x line, so any point sitting above it means the condition produced more uncertainty than Normal did
all_vals = pd.concat([prompt_df["normal_entropy"],
                      prompt_df["misrouted_entropy"],
                      prompt_df["top1_entropy"]])
lim = (all_vals.min() - 0.05, all_vals.max() + 0.05)
ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.5, label="y = x (baseline)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_title("Normal Entropy vs Random & Top-1 per Prompt\n"
             "(points above diagonal = higher uncertainty than baseline)",
             fontsize=TITLE_SIZE - 2, fontweight="bold")
ax.set_xlabel("Normal Entropy (nats)", fontsize=LABEL_SIZE)
ax.set_ylabel("Condition Entropy (nats)", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE - 2, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/scatter_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/scatter_entropy.png")
plt.close()


# Histogram of entropy values for each of the three conditions
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle("Histogram of Semantic Entropy Values Across All Prompts",
             fontsize=TITLE_SIZE, fontweight="bold")

# Use the same bin edges across all three histograms so they can be compared directly
max_val = prompt_df[["normal_entropy", "misrouted_entropy", "top1_entropy"]].max().max()
bins    = np.linspace(0, max_val + 0.1, 25)

for ax, (condition, label) in zip(axes, condition_order):
    vals = prompt_df[f"{condition}_entropy"]
    ax.hist(vals, bins=bins, color=colors[condition], alpha=0.85, edgecolor="white")
    ax.set_title(label, fontsize=TITLE_SIZE - 2, fontweight="bold")
    ax.set_xlabel("Semantic Entropy (nats)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Prompts" if ax == axes[0] else "", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=2,
               label=f"Mean = {vals.mean():.3f}")
    ax.legend(fontsize=LEGEND_SIZE - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/histogram_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/histogram_entropy.png")
plt.close()


# Read the original prompts back in so we can attach the actual text to each row in the appendix tables
prompts_by_category = {}
with open("results/normal.jsonl") as f:
    for line in f:
        item = json.loads(line)
        cat  = item["category"]
        prompts_by_category.setdefault(cat, []).append(item["prompt"])

prompt_df = prompt_df.copy()
prompt_df["prompt_text"] = ""
for cat, prompts in prompts_by_category.items():
    idx = prompt_df[prompt_df["category"] == cat].index
    for i, row_idx in enumerate(idx):
        if i < len(prompts):
            prompt_df.at[row_idx, "prompt_text"] = prompts[i]

import os

# A separate heatmap per category, showing the entropy difference for each prompt against the Normal condition
# Find the biggest absolute difference across both pairs so all three heatmaps can share one colour scale
max_diff = max(
    (prompt_df["misrouted_entropy"] - prompt_df["normal_entropy"]).abs().max(),
    (prompt_df["top1_entropy"]      - prompt_df["normal_entropy"]).abs().max()
)

for cat, cat_label in zip(categories, cat_labels):
    subset = prompt_df[prompt_df["category"] == cat].copy()
    subset = subset.sort_values("normal_entropy").reset_index(drop=True)

    # Two columns: how much Random shifted from Normal, and how much Top-1 shifted from Normal
    diff_data = np.column_stack([
        subset["misrouted_entropy"].values - subset["normal_entropy"].values,
        subset["top1_entropy"].values      - subset["normal_entropy"].values,
    ])

    # Scale the figure height with the number of prompts so the rows don't get squashed
    fig_height = max(8, 0.32 * len(subset) + 2.2)
    fig, ax = plt.subplots(figsize=(5.8, fig_height))

    im = ax.imshow(
        diff_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_diff,
        vmax=max_diff,
        interpolation="nearest"
    )

    ax.set_title(
        f"Entropy Change Relative to Normal Routing: {cat_label}",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\n− Normal", "Top-1\n− Normal"], fontsize=11)

    # Show short P1, P2... labels on the axis instead of full prompt text so the rows stay readable
    prompt_ids = [f"P{i+1}" for i in range(len(subset))]
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(prompt_ids, fontsize=9)

    ax.set_ylabel("Prompt ID (sorted by normal entropy)", fontsize=11)
    ax.tick_params(axis="y", pad=3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#666666")

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Entropy Change (nats)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    out_path = f"results/appendix_diff_heatmap_{cat}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

# Lookup table per category so the P1, P2... labels in the heatmaps can be matched back to the actual prompts
for cat, cat_label in zip(categories, cat_labels):
    subset = prompt_df[prompt_df["category"] == cat].copy()
    subset = subset.sort_values("normal_entropy").reset_index(drop=True)

    fig_height = max(6, 0.28 * len(subset) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table_rows = []
    for i, prompt in enumerate(subset["prompt_text"].values, start=1):
        table_rows.append([f"P{i}", prompt])

    table = ax.table(
        cellText=table_rows,
        colLabels=["Prompt ID", "Prompt"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.12, 0.88]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for j in range(2):
        table[0, j].set_facecolor("#1e3a5f")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle(f"Prompt Key: {cat_label}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = f"results/prompt_key_{cat}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

# Violin plot per category, with the shape of each violin showing how the entropy values are spread out
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
fig.suptitle("Violin Plot of Semantic Entropy Distribution by Category and Routing Condition",
             fontsize=15, fontweight="bold")

for ax, cat, cat_label in zip(axes, categories, cat_labels):
    subset = prompt_df[prompt_df["category"] == cat]
    data   = [subset["normal_entropy"].values,
              subset["misrouted_entropy"].values,
              subset["top1_entropy"].values]

    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True,
                          showextrema=True)

    # Colour the violin bodies to match their routing condition
    for pc, (cond, _) in zip(parts["bodies"], condition_order):
        pc.set_facecolor(colors[cond])
        pc.set_alpha(0.75)

    # Make the median line and the min/max bars solid black so they read clearly against the coloured bodies
    for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[part].set_color("black")
        parts[part].set_linewidth(2)

    ax.set_title(cat_label, fontsize=14, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Normal", "Random", "Top-1"], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Semantic Entropy (nats)" if ax == axes[0] else "", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

legend_elements = [Patch(facecolor=colors[c], alpha=0.75, label=l)
                   for c, l in condition_order]
fig.legend(handles=legend_elements, fontsize=12, loc="lower center",
           ncol=3, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
plt.savefig("results/appendix_violin.png", dpi=150, bbox_inches="tight")
print("Saved: results/appendix_violin.png")
plt.close()


# One scatter per condition, plotting cluster count against entropy to show how the two relate
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cluster Count vs Semantic Entropy per Prompt",
             fontsize=15, fontweight="bold")

for ax, (condition, label) in zip(axes, condition_order):
    ax.scatter(prompt_df[f"{condition}_clusters"],
               prompt_df[f"{condition}_entropy"],
               color=colors[condition], alpha=0.6, s=60, edgecolors="none")
    ax.set_title(label, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel("Semantic Entropy (nats)" if ax == axes[0] else "", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/appendix_cluster_vs_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/appendix_cluster_vs_entropy.png")
plt.close()


print("\nAll visualisations saved to results/")
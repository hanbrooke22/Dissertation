import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    "Misrouted (mean ± std)",
    "Top-1 (mean ± std)",
    "Diff: Misrouted − Normal",
    "Diff: Top-1 − Normal",
]


def style_table(table, n_cols, n_data_rows):
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


# ── Table 1: Entropy ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
fig.suptitle("Table 1: Mean Semantic Entropy by Category and Routing Condition",
             fontsize=12, fontweight="bold", y=0.98)

table_data = []
for condition in ["normal", "misrouted", "top1"]:
    row = []
    for cat in categories:
        r    = category_df[category_df["category"] == cat].iloc[0]
        mean = r[f"{condition}_mean_entropy"]
        std  = r[f"{condition}_std_entropy"]
        row.append(f"{mean:.4f} ± {std:.4f}")
    table_data.append(row)

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


# ── Table 2: Clusters ──────────────────────────────────────────────────────────
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


# ── Bar Chart 1: Entropy ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
x     = np.arange(len(categories))
width = 0.25

for i, (condition, label) in enumerate(
        [("normal", "Normal"), ("misrouted", "Misrouted"), ("top1", "Top-1")]):
    means = [category_df[category_df["category"] == cat].iloc[0][f"{condition}_mean_entropy"]
             for cat in categories]
    stds  = [category_df[category_df["category"] == cat].iloc[0][f"{condition}_std_entropy"]
             for cat in categories]
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, yerr=stds, capsize=5,
           label=label, color=colors[condition], alpha=0.85)

ax.set_title("Mean Semantic Entropy by Category: All Routing Conditions",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Semantic Entropy (nats)", fontsize=11)
ax.set_xlabel("Prompt Category", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=11)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/bar_chart_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/bar_chart_entropy.png")
plt.close()


# ── Bar Chart 2: Clusters ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

for i, (condition, label) in enumerate(
        [("normal", "Normal"), ("misrouted", "Misrouted"), ("top1", "Top-1")]):
    means = [prompt_df[prompt_df["category"] == cat][f"{condition}_clusters"].mean()
             for cat in categories]
    stds  = [prompt_df[prompt_df["category"] == cat][f"{condition}_clusters"].std()
             for cat in categories]
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, yerr=stds, capsize=5,
           label=label, color=colors[condition], alpha=0.85)

ax.set_title("Mean Cluster Count by Category: All Routing Conditions",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Mean Number of Clusters", fontsize=11)
ax.set_xlabel("Prompt Category", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=11)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/bar_chart_clusters.png", dpi=150, bbox_inches="tight")
print("Saved: results/bar_chart_clusters.png")
plt.close()


# ── Box Plots: Entropy distribution per condition per category ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
fig.suptitle("Distribution of Semantic Entropy by Category and Routing Condition",
             fontsize=13, fontweight="bold")

for ax, cat, cat_label in zip(axes, categories, cat_labels):
    subset = prompt_df[prompt_df["category"] == cat]
    data   = [subset["normal_entropy"].values,
              subset["misrouted_entropy"].values,
              subset["top1_entropy"].values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], [colors["normal"], colors["misrouted"], colors["top1"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(cat_label, fontsize=12, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Normal", "Misrouted", "Top-1"], fontsize=10)
    ax.set_ylabel("Semantic Entropy (nats)" if ax == axes[0] else "", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/box_plots_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/box_plots_entropy.png")
plt.close()


# ── Scatter: Normal entropy vs Misrouted and Top-1 per prompt ─────────────────
# Each point is one prompt; points above the diagonal have higher entropy
# than baseline, indicating greater response variation in that condition
fig, ax = plt.subplots(figsize=(8, 8))

scatter_styles = {
    ("misrouted", "unanswerable"): {"color": "#2563EB", "marker": "o", "label": "Misrouted – Unanswerable"},
    ("misrouted", "open_ended"):   {"color": "#16A34A", "marker": "o", "label": "Misrouted – Open Ended"},
    ("misrouted", "factual"):      {"color": "#9333EA", "marker": "o", "label": "Misrouted – Factual"},
    ("top1",      "unanswerable"): {"color": "#DC2626", "marker": "^", "label": "Top-1 – Unanswerable"},
    ("top1",      "open_ended"):   {"color": "#F59E0B", "marker": "^", "label": "Top-1 – Open Ended"},
    ("top1",      "factual"):      {"color": "#0891B2", "marker": "^", "label": "Top-1 – Factual"},
}

for (condition, cat), style in scatter_styles.items():
    subset = prompt_df[prompt_df["category"] == cat]
    ax.scatter(subset["normal_entropy"], subset[f"{condition}_entropy"],
               color=style["color"], marker=style["marker"],
               alpha=0.75, s=60, label=style["label"])

all_vals = pd.concat([prompt_df["normal_entropy"],
                      prompt_df["misrouted_entropy"],
                      prompt_df["top1_entropy"]])
lim = (all_vals.min() - 0.05, all_vals.max() + 0.05)
ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="y = x (baseline)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_title("Normal Entropy vs Misrouted & Top-1 per Prompt\n(points above diagonal = higher uncertainty than baseline)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Normal Entropy (nats)", fontsize=11)
ax.set_ylabel("Condition Entropy (nats)", fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/scatter_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/scatter_entropy.png")
plt.close()


# ── Histogram: Entropy distribution across all prompts ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Histogram of Semantic Entropy Values Across All Prompts",
             fontsize=13, fontweight="bold")

max_val = prompt_df[["normal_entropy", "misrouted_entropy", "top1_entropy"]].max().max()
bins    = np.linspace(0, max_val + 0.1, 25)

for ax, (condition, label) in zip(axes, [("normal", "Normal"), ("misrouted", "Misrouted"), ("top1", "Top-1")]):
    vals = prompt_df[f"{condition}_entropy"]
    ax.hist(vals, bins=bins, color=colors[condition], alpha=0.85, edgecolor="white")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Semantic Entropy (nats)", fontsize=10)
    ax.set_ylabel("Number of Prompts" if ax == axes[0] else "", fontsize=10)
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {vals.mean():.3f}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/histogram_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/histogram_entropy.png")
plt.close()

print("\nAll visualisations saved to results/")
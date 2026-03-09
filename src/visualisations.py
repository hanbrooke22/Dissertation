import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prompt_df   = pd.read_csv("results/per_prompt_comparison.csv")
category_df = pd.read_csv("results/per_category_comparison.csv")

categories = ["unanswerable", "open_ended", "factual"]
cat_labels = ["Unanswerable", "Open Ended", "Factual"]

colors = {
    "normal":  "#2563EB",
    "random":  "#F59E0B",
    "bottom":  "#DC2626",
}

row_labels = [
    "Normal (mean ± std)",
    "Random (mean ± std)",
    "Bottom-4 (mean ± std)",
    "Diff: Random − Normal",
    "Diff: Bottom-4 − Normal",
]


# ── Helper: style a table ──────────────────────────────────────────────────────
def style_table(table, n_cols, n_data_rows):
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    # Header row
    for j in range(n_cols):
        table[0, j].set_facecolor("#1e3a5f")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Diff rows
    for j in range(n_cols):
        table[n_data_rows - 1, j].set_facecolor("#fff7f7")
        table[n_data_rows - 2, j].set_facecolor("#fff7f7")
    # Row label column
    for i in range(1, n_data_rows + 1):
        table[i, -1].set_facecolor("#f0f4ff")
        table[i, -1].set_text_props(fontweight="bold")


# ── Table 1: Entropy ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
fig.suptitle("Table 1: Mean Semantic Entropy by Category and Routing Condition",
             fontsize=12, fontweight="bold", y=0.98)

table_data = []
for condition in ["normal", "random", "bottom"]:
    row = []
    for cat in categories:
        r    = category_df[category_df["category"] == cat].iloc[0]
        mean = r[f"{condition}_mean_entropy"]
        std  = r[f"{condition}_std_entropy"]
        row.append(f"{mean:.4f} ± {std:.4f}")
    table_data.append(row)

for diff_pair in [("random", "normal"), ("bottom", "normal")]:
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
for condition in ["normal", "random", "bottom"]:
    row = []
    for cat in categories:
        vals = prompt_df[prompt_df["category"] == cat][f"{condition}_clusters"]
        row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")
    table_data2.append(row)

for diff_pair in [("random", "normal"), ("bottom", "normal")]:
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


# ── Bar chart helper ───────────────────────────────────────────────────────────
def annotate_bar(ax, bar, val, y_max, label_prefix, color):
    x_right = bar.get_x() + bar.get_width() + 0.01
    ax.text(x_right, val, f"{label_prefix}={val:.3f}",
            ha="left", va="center", fontsize=7, color=color)


# ── Bar Chart 1: Entropy ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

x     = np.arange(len(categories))
width = 0.25

for i, (condition, label) in enumerate(
        [("normal", "Normal"), ("random", "Random"), ("bottom", "Bottom-4")]):
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
        [("normal", "Normal"), ("random", "Random"), ("bottom", "Bottom-4")]):
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
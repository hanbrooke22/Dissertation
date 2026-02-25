import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prompt_df = pd.read_csv("results/per_prompt_comparison.csv")
category_df = pd.read_csv("results/per_category_comparison.csv")

CATEGORIES = ["unanswerable", "open_ended", "single_answer"]
CAT_LABELS = ["Unanswerable", "Open Ended", "Single Answer"]
COLORS = {"normal": "#2563EB", "misrouted": "#DC2626"}
row_labels = ["Normal (mean ± std)", "Misrouted (mean ± std)", "Difference"]


fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
fig.suptitle("Table 1: Mean Semantic Entropy by Category and Routing Condition",
             fontsize=12, fontweight="bold", y=0.98)

table_data = []
for condition in ["normal", "misrouted"]:
    row = []
    for cat in CATEGORIES:
        r = category_df[category_df["category"] == cat].iloc[0]
        mean = r[f"{condition}_mean_entropy"]
        std  = r[f"{condition}_std_entropy"]
        row.append(f"{mean:.4f} ± {std:.4f}")
    table_data.append(row)

diff_row = []
for cat in CATEGORIES:
    diff = category_df[category_df["category"] == cat].iloc[0]["mean_entropy_diff"]
    sign = "+" if diff > 0 else ""
    diff_row.append(f"{sign}{diff:.4f}")
table_data.append(diff_row)

table = ax.table(cellText=table_data, rowLabels=row_labels,
                 colLabels=CAT_LABELS, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

for j in range(len(CAT_LABELS)):
    table[0, j].set_facecolor("#1e3a5f")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(row_labels)):
    table[i+1, -1].set_facecolor("#f0f4ff")
    table[i+1, -1].set_text_props(fontweight="bold")
for j in range(len(CAT_LABELS)):
    table[3, j].set_facecolor("#fff7f7")

plt.tight_layout()
plt.savefig("results/table_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/table_entropy.png")
plt.close()


fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
fig.suptitle("Table 2: Mean Cluster Count by Category and Routing Condition",
             fontsize=12, fontweight="bold", y=0.98)

table_data2 = []
for condition in ["normal", "misrouted"]:
    row = []
    for cat in CATEGORIES:
        vals = prompt_df[prompt_df["category"] == cat][f"{condition}_clusters"]
        row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")
    table_data2.append(row)

diff_row2 = []
for cat in CATEGORIES:
    subset = prompt_df[prompt_df["category"] == cat]
    diff = subset["misrouted_clusters"].mean() - subset["normal_clusters"].mean()
    sign = "+" if diff > 0 else ""
    diff_row2.append(f"{sign}{diff:.2f}")
table_data2.append(diff_row2)

table2 = ax.table(cellText=table_data2, rowLabels=row_labels,
                  colLabels=CAT_LABELS, cellLoc="center", loc="center")
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1.2, 2.0)

for j in range(len(CAT_LABELS)):
    table2[0, j].set_facecolor("#1e3a5f")
    table2[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(row_labels)):
    table2[i+1, -1].set_facecolor("#f0f4ff")
    table2[i+1, -1].set_text_props(fontweight="bold")
for j in range(len(CAT_LABELS)):
    table2[3, j].set_facecolor("#fff7f7")

plt.tight_layout()
plt.savefig("results/table_clusters.png", dpi=150, bbox_inches="tight")
print("Saved: results/table_clusters.png")
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(CATEGORIES))
width = 0.35

normal_means    = [category_df[category_df["category"] == cat].iloc[0]["normal_mean_entropy"]    for cat in CATEGORIES]
normal_stds     = [category_df[category_df["category"] == cat].iloc[0]["normal_std_entropy"]     for cat in CATEGORIES]
misrouted_means = [category_df[category_df["category"] == cat].iloc[0]["misrouted_mean_entropy"] for cat in CATEGORIES]
misrouted_stds  = [category_df[category_df["category"] == cat].iloc[0]["misrouted_std_entropy"]  for cat in CATEGORIES]

bars1 = ax.bar(x - width/2, normal_means, width, yerr=normal_stds, capsize=5,
               label="Normal Routing", color=COLORS["normal"], alpha=0.85)
bars2 = ax.bar(x + width/2, misrouted_means, width, yerr=misrouted_stds, capsize=5,
               label="Misrouted", color=COLORS["misrouted"], alpha=0.85)

ax.set_title("Mean Semantic Entropy by Category: Normal vs Misrouted Routing",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Semantic Entropy (nats)", fontsize=11)
ax.set_xlabel("Prompt Category", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(CAT_LABELS, fontsize=11)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(bottom=0)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=COLORS["normal"])
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=COLORS["misrouted"])

plt.tight_layout()
plt.savefig("results/bar_chart_entropy.png", dpi=150, bbox_inches="tight")
print("Saved: results/bar_chart_entropy.png")
plt.close()

print("\nAll visualisations complete.")
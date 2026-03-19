import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from collections import defaultdict

per_prompt_path = "results/per_prompt_comparison.json"


def sig(p: float) -> str:
    # Convert p-value to significance stars for display
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_shapiro_on_differences(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Shapiro-Wilk normality test applied to the difference scores (a - b),
    as required by the paired t-test normality assumption.
    Returns W statistic, p-value, sig, and normality verdict.
    """
    diff = a - b
    if len(diff) < 3 or np.all(diff == diff[0]):
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a", "normal": "n/a"}
    w, p = stats.shapiro(diff)
    normal = "yes" if p >= 0.05 else "no"
    return {"W": float(w), "p": float(p), "sig": sig(p), "normal": normal}


def run_paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test between two arrays."""
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "sig": sig(p)}


def run_wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test between two arrays."""
    diff = a - b
    if np.all(diff == 0):
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a"}
    try:
        stat, p = stats.wilcoxon(a, b)
        return {"W": float(stat), "p": float(p), "sig": sig(p)}
    except ValueError:
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a"}


def run_statistical_tests(normal: np.ndarray, misrouted: np.ndarray, top1: np.ndarray, label: str) -> dict:
    """
    For each pairwise comparison:
      1. Compute difference scores
      2. Run Shapiro-Wilk on the differences
      3. If normally distributed (p >= 0.05): use paired t-test
         If not normally distributed (p < 0.05): use Wilcoxon signed-rank test
    """
    pairs = [
        ("normal_vs_misrouted", normal,   misrouted, "Normal vs Misrouted"),
        ("normal_vs_top1",      normal,   top1,      "Normal vs Top-1"),
        ("misrouted_vs_top1",   misrouted, top1,     "Misrouted vs Top-1"),
    ]

    print(f"\n{'='*65}")
    print(f"STATISTICAL TESTS — {label}")
    print(f"{'='*65}")
    print(f"N: {len(normal)}\n")

    print(f"{'Condition':<12} {'Mean':>8} {'Std':>8}")
    print("-"*32)
    print(f"{'Normal':<12} {normal.mean():>8.4f} {normal.std():>8.4f}")
    print(f"{'Misrouted':<12} {misrouted.mean():>8.4f} {misrouted.std():>8.4f}")
    print(f"{'Top-1':<12} {top1.mean():>8.4f} {top1.std():>8.4f}")

    shapiro_results = {}
    test_results    = {}
    test_used       = {}

    print(f"\n{'Comparison':<28} {'SW-W':>8} {'SW-p':>8} {'Normal?':>8} {'Test Used':>14}")
    print("-"*70)

    for key, a, b, comp_label in pairs:
        sw = run_shapiro_on_differences(a, b)
        shapiro_results[key] = sw
        is_normal = sw["normal"] == "yes"

        # Select test based on normality of difference scores
        if is_normal:
            result = run_paired_ttest(a, b)
            test_used[key] = "paired t-test"
        else:
            result = run_wilcoxon_pair(a, b)
            test_used[key] = "wilcoxon"

        test_results[key] = result

        sw_W = f"{sw['W']:.4f}" if sw['W'] == sw['W'] else "n/a"
        sw_p = f"{sw['p']:.4f}" if sw['p'] == sw['p'] else "n/a"
        print(f"{comp_label:<28} {sw_W:>8} {sw_p:>8} {sw['normal']:>8} {test_used[key]:>14}")

    print(f"\n{'Comparison':<28} {'Statistic':>10} {'p':>10} {'sig':>6} {'Test':>14}")
    print("-"*72)
    for key, a, b, comp_label in pairs:
        result = test_results[key]
        test   = test_used[key]
        if test == "paired t-test":
            stat_str = f"t={result['t']:.4f}"
        else:
            stat_str = f"W={result['W']:.1f}" if result['W'] == result['W'] else "W=n/a"
        p_str = f"{result['p']:.4f}" if result['p'] == result['p'] else "n/a"
        print(f"{comp_label:<28} {stat_str:>10} {p_str:>10} {result['sig']:>6} {test:>14}")

    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")

    return {
        "n": len(normal),
        "means": {
            "normal":   float(normal.mean()),
            "misrouted": float(misrouted.mean()),
            "top1":     float(top1.mean()),
        },
        "stds": {
            "normal":   float(normal.std()),
            "misrouted": float(misrouted.std()),
            "top1":     float(top1.std()),
        },
        "shapiro_wilk_differences": shapiro_results,
        "test_used":  test_used,
        "test_results": test_results,
    }


# Load per-prompt entropy scores 
with open(per_prompt_path, "r") as f:
    prompt_rows = json.load(f)

normal_all    = np.array([r["normal_entropy"]    for r in prompt_rows])
misrouted_all = np.array([r["misrouted_entropy"] for r in prompt_rows])
top1_all      = np.array([r["top1_entropy"]      for r in prompt_rows])

# Overall tests
overall_results = run_statistical_tests(normal_all, misrouted_all, top1_all, "OVERALL (n=81)")

# Per-category statistical tests 
cat_data = defaultdict(lambda: {"normal": [], "misrouted": [], "top1": []})
for row in prompt_rows:
    cat_data[row["category"]]["normal"].append(row["normal_entropy"])
    cat_data[row["category"]]["misrouted"].append(row["misrouted_entropy"])
    cat_data[row["category"]]["top1"].append(row["top1_entropy"])

category_results = {}
for category, vals in cat_data.items():
    n_arr = np.array(vals["normal"])
    r_arr = np.array(vals["misrouted"])
    t_arr = np.array(vals["top1"])
    category_results[category] = run_statistical_tests(
        n_arr, r_arr, t_arr, f"{category.upper()} (n={len(n_arr)})"
    )

# Save results to JSON 
output = {"overall": overall_results, "by_category": category_results}
with open("results/statistical_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to results/statistical_results.json")


# Table visualisation helpers
HEADER_COLOR  = "#1e3a5f"
SUBHEAD_COLOR = "#2d5986"
ALT_COLOR     = "#f0f4ff"
WHITE         = "#ffffff"

def fmt_p(p):
    # Format p-values for display, flagging values below 0.001
    if p != p: return "n/a"
    if p < 0.001: return "< 0.001"
    return f"{p:.4f}"

def fmt_val(v):
    if v != v: return "n/a"
    return f"{v:.4f}"

def make_header_cell(ax, text, x, y, w, h, fontsize=10):
    # Draw a dark header cell with white bold te
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", facecolor=HEADER_COLOR, edgecolor="white", linewidth=0.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold")

def make_cell(ax, text, x, y, w, h, bg=WHITE, color="black", fontsize=9, bold=False):
    # Draw a standard table cell with configurable background and text colour
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", facecolor=bg, edgecolor="#cccccc", linewidth=0.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold" if bold else "normal")

def sig_color(s):
    # Map significance level to colour for visual emphasis in tables
    if s == "***": return "#16a34a"
    if s == "**":  return "#ca8a04"
    if s == "*":   return "#ea580c"
    if s == "ns":  return "#6b7280"
    return "#6b7280"


# Table A: Shapiro-Wilk on difference scores
def draw_shapiro_table(data: dict, cat_order: list, filename: str):
    """
    Rows: each pairwise comparison
    Cols: Overall + each category
    Shows W, p, normality verdict, and which test was selected
    """
    comparisons  = ["normal_vs_misrouted", "normal_vs_top1", "misrouted_vs_top1"]
    comp_labels  = ["Normal vs Misrouted", "Normal vs Top-1", "Misrouted vs Top-1"]
    col_labels   = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(comparisons)
    n_cols = len(col_labels)
    col_w  = 2.0
    row_h  = 0.8
    lbl_w  = 2.0
    fig_w  = lbl_w + n_cols * col_w + 0.4
    fig_h  = row_h * (n_rows + 1) + 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Shapiro-Wilk Test on Difference Scores — with Selected Test",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Comparison", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    col_keys = ["overall"] + cat_order
    all_data = {"overall": data["overall"]}
    for cat in cat_order:
        all_data[cat] = data[cat]

    for i, (comp, comp_label) in enumerate(zip(comparisons, comp_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = ALT_COLOR if i % 2 == 0 else WHITE
        make_cell(ax, comp_label, 0, y, lbl_w, row_h,
                  bg=SUBHEAD_COLOR, color="white", bold=True, fontsize=9)
        for j, col_key in enumerate(col_keys):
            sw      = all_data[col_key]["shapiro_wilk_differences"][comp]
            test    = all_data[col_key]["test_used"][comp]
            W       = fmt_val(sw["W"])
            p       = fmt_p(sw["p"])
            verdict = sw["normal"]
            test_short = "t-test" if test == "paired t-test" else "Wilcoxon"
            cell_text = f"W={W}\np={p}\nNormal: {verdict}\n→ {test_short}"
            color = "#16a34a" if verdict == "yes" else "#dc2626"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h,
                      bg=bg, color=color, fontsize=7.5)

    ax.text(0, -0.18,
            "Normal: p ≥ 0.05 → paired t-test used  |  Not normal: p < 0.05 → Wilcoxon signed-rank used",
            fontsize=7.5, color="#444444", transform=ax.transData)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# Table B: Main test results (t-test or Wilcoxon per cell)
def draw_test_results_table(data: dict, cat_order: list, filename: str):
    """
    Shows the result of whichever test was selected per comparison per category.
    """
    comparisons  = ["normal_vs_misrouted", "normal_vs_top1", "misrouted_vs_top1"]
    comp_labels  = ["Normal vs Misrouted", "Normal vs Top-1", "Misrouted vs Top-1"]
    col_labels   = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(comparisons)
    n_cols = len(col_labels)
    col_w  = 2.0
    row_h  = 0.8
    lbl_w  = 2.0
    fig_w  = lbl_w + n_cols * col_w + 0.4
    fig_h  = row_h * (n_rows + 1) + 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Statistical Test Results — Paired t-test or Wilcoxon (conditional on normality)",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Comparison", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    col_keys = ["overall"] + cat_order
    all_data = {"overall": data["overall"]}
    for cat in cat_order:
        all_data[cat] = data[cat]

    for i, (comp, comp_label) in enumerate(zip(comparisons, comp_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = ALT_COLOR if i % 2 == 0 else WHITE
        make_cell(ax, comp_label, 0, y, lbl_w, row_h,
                  bg=SUBHEAD_COLOR, color="white", bold=True, fontsize=9)
        for j, col_key in enumerate(col_keys):
            result     = all_data[col_key]["test_results"][comp]
            test       = all_data[col_key]["test_used"][comp]
            test_short = "t-test" if test == "paired t-test" else "Wilcoxon"
            if test == "paired t-test":
                stat_str = f"t = {result['t']:.3f}" if result['t'] == result['t'] else "t = n/a"
            else:
                stat_str = f"W = {result['W']:.1f}" if result['W'] == result['W'] else "W = n/a"
            p   = fmt_p(result["p"])
            s   = result["sig"]
            cell_text = f"{stat_str}\np = {p}\n{s}\n[{test_short}]"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h,
                      bg=bg, color=sig_color(s), fontsize=7.5, bold=(s not in ("ns", "n/a")))

    ax.text(0, -0.18,
            "Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
            fontsize=7.5, color="#444444", transform=ax.transData)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# Table C: Descriptive stats 
def draw_descriptive_table(data: dict, cat_order: list, filename: str):
    # Display mean ± SD entropy per condition per category
    conditions  = ["normal", "misrouted", "top1"]
    cond_labels = ["Normal", "Misrouted", "Top-1"]
    col_labels  = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(conditions)
    n_cols = len(col_labels)
    col_w  = 2.0
    row_h  = 0.7
    lbl_w  = 1.4
    fig_w  = lbl_w + n_cols * col_w + 0.4
    fig_h  = row_h * (n_rows + 1) + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Descriptive Statistics — Mean ± SD Semantic Entropy by Condition & Category",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Condition", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    col_keys = ["overall"] + cat_order
    all_data = {"overall": data["overall"]}
    for cat in cat_order:
        all_data[cat] = data[cat]

    colors_cond = {"normal": "#dbeafe", "misrouted": "#fef3c7", "top1": "#fee2e2"}

    for i, (cond, cond_label) in enumerate(zip(conditions, cond_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = colors_cond[cond]
        make_cell(ax, cond_label, 0, y, lbl_w, row_h,
                  bg=SUBHEAD_COLOR, color="white", bold=True)
        for j, col_key in enumerate(col_keys):
            mean = all_data[col_key]["means"][cond]
            std  = all_data[col_key]["stds"][cond]
            cell_text = f"{mean:.4f}\n± {std:.4f}"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h, bg=bg, fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# Generate all three tables 
cat_order = ["unanswerable", "open_ended", "factual"]

combined = {
    "overall":      overall_results,
    "unanswerable": category_results.get("unanswerable", {}),
    "open_ended":   category_results.get("open_ended",   {}),
    "factual":      category_results.get("factual",      {}),
}

draw_shapiro_table(combined,      cat_order, "results/table_shapiro_wilk.png")
draw_test_results_table(combined, cat_order, "results/table_test_results.png")
draw_descriptive_table(combined,  cat_order, "results/table_descriptive.png")

print("\nAll tables saved to results/")
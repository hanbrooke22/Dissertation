import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from collections import defaultdict

per_prompt_path = "results/per_prompt_comparison.json"


def sig(p: float) -> str:
    # Turn a p-value into the usual star notation
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired samples.
    d = mean(differences) / std(differences, ddof=1)
    Interpretation: small=0.2, medium=0.5, large=0.8
    """
    diff = a - b
    if np.std(diff, ddof=1) == 0:
        return float("nan")
    return float(np.mean(diff) / np.std(diff, ddof=1))


def wilcoxon_r(a: np.ndarray, b: np.ndarray) -> float:
    """
    Effect size r for Wilcoxon signed-rank test.
    r = Z / sqrt(N), where Z is derived from the Wilcoxon statistic.
    When W=0.0, effect is at maximum so r approaches 1.0.
    Interpretation: small=0.1, medium=0.3, large=0.5
    """
    diff = a - b
    if np.all(diff == 0):
        return float("nan")
    try:
        # Run the Wilcoxon test to grab its W statistic
        stat, p = stats.wilcoxon(a, b)
        n = len(a)
        # Work out a z-score from W using a normal approximation, based on what W's mean and spread would look like if there were no real effect
        mu_w = n * (n + 1) / 4
        sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = abs((stat - mu_w) / sigma_w)
        r = z / np.sqrt(n)
        return float(r)
    except ValueError:
        return float("nan")


def effect_label(d: float, test: str) -> str:
    """Return a human-readable effect size label."""
    # NaN doesn't equal itself, so this catches it
    if d != d:
        return "n/a"
    d = abs(d)
    if test == "paired t-test":
        # Standard cut-offs for Cohen's d
        if d >= 0.8:  return "large"
        if d >= 0.5:  return "medium"
        if d >= 0.2:  return "small"
        return "negligible"
    else:
        # Standard cut-offs for the Wilcoxon r effect size
        if d >= 0.5:  return "large"
        if d >= 0.3:  return "medium"
        if d >= 0.1:  return "small"
        return "negligible"


def run_shapiro_on_differences(a: np.ndarray, b: np.ndarray) -> dict:
    # Run the Shapiro-Wilk test on the paired differences to see if they're roughly normal
    diff = a - b
    if len(diff) < 3 or np.all(diff == diff[0]):
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a", "normal": "n/a"}
    w, p = stats.shapiro(diff)
    normal = "yes" if p >= 0.05 else "no"
    return {"W": float(w), "p": float(p), "sig": sig(p), "normal": normal}


def run_paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    # Run a paired t-test and report Cohen's d as the effect size
    t, p = stats.ttest_rel(a, b)
    d = cohens_d_paired(a, b)
    return {"t": float(t), "p": float(p), "sig": sig(p),
            "effect_size": d, "effect_label": effect_label(d, "paired t-test"),
            "effect_type": "Cohen's d"}


def run_wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> dict:
    # Run the Wilcoxon signed-rank test as the fallback when the differences aren't normal
    diff = a - b
    if np.all(diff == 0):
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a",
                "effect_size": float("nan"), "effect_label": "n/a",
                "effect_type": "r"}
    try:
        stat, p = stats.wilcoxon(a, b)
        r = wilcoxon_r(a, b)
        return {"W": float(stat), "p": float(p), "sig": sig(p),
                "effect_size": r, "effect_label": effect_label(r, "wilcoxon"),
                "effect_type": "r"}
    except ValueError:
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a",
                "effect_size": float("nan"), "effect_label": "n/a",
                "effect_type": "r"}


def run_statistical_tests(normal: np.ndarray, misrouted: np.ndarray, top1: np.ndarray, label: str) -> dict:
    # Run all three pairwise comparisons, picking the right test for each based on the Shapiro-Wilk result
    pairs = [
        ("normal_vs_misrouted", normal,    misrouted, "Normal vs Misrouted"),
        ("normal_vs_top1",      normal,    top1,      "Normal vs Top-1"),
        ("misrouted_vs_top1",   misrouted, top1,      "Misrouted vs Top-1"),
    ]

    print(f"\n{'='*75}")
    print(f"STATISTICAL TESTS — {label}")
    print(f"{'='*75}")
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
        # Check for normality first, then go with the t-test if it passes or the Wilcoxon if it doesn't
        sw = run_shapiro_on_differences(a, b)
        shapiro_results[key] = sw
        is_normal = sw["normal"] == "yes"

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

    # Print the test stats, p-values, and effect sizes for each comparison
    print(f"\n{'Comparison':<28} {'Statistic':>10} {'p':>10} {'sig':>6} {'Effect':>10} {'Size':>8} {'Magnitude':>12}")
    print("-"*90)
    for key, a, b, comp_label in pairs:
        result = test_results[key]
        test   = test_used[key]
        if test == "paired t-test":
            stat_str = f"t={result['t']:.4f}"
        else:
            stat_str = f"W={result['W']:.1f}" if result['W'] == result['W'] else "W=n/a"
        p_str  = f"{result['p']:.4f}" if result['p'] == result['p'] else "n/a"
        es     = result['effect_size']
        es_str = f"{es:.4f}" if es == es else "n/a"
        print(f"{comp_label:<28} {stat_str:>10} {p_str:>10} {result['sig']:>6} "
              f"{result['effect_type']:>10} {es_str:>8} {result['effect_label']:>12}")

    print("\nEffect size benchmarks:")
    print("  Cohen's d — small: 0.2, medium: 0.5, large: 0.8")
    print("  r         — small: 0.1, medium: 0.3, large: 0.5")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")

    return {
        "n": len(normal),
        "means":  {"normal": float(normal.mean()), "misrouted": float(misrouted.mean()), "top1": float(top1.mean())},
        "stds":   {"normal": float(normal.std()),  "misrouted": float(misrouted.std()),  "top1": float(top1.std())},
        "shapiro_wilk_differences": shapiro_results,
        "test_used":    test_used,
        "test_results": test_results,
    }


# Load the per-prompt entropy scores from the previous step
with open(per_prompt_path, "r") as f:
    prompt_rows = json.load(f)

normal_all    = np.array([r["normal_entropy"]    for r in prompt_rows])
misrouted_all = np.array([r["misrouted_entropy"] for r in prompt_rows])
top1_all      = np.array([r["top1_entropy"]      for r in prompt_rows])

# Run the full set of tests on the whole dataset before breaking things down by category
overall_results = run_statistical_tests(normal_all, misrouted_all, top1_all, "OVERALL (n=81)")

# Split the entropy scores up by category so the same tests can be run within each one
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

# Save the overall and per-category results into one JSON file for later
output = {"overall": overall_results, "by_category": category_results}
with open("results/statistical_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to results/statistical_results.json")


# Shared colours and cell-drawing helpers used by all three tables below
HEADER_COLOR  = "#1e3a5f"
SUBHEAD_COLOR = "#2d5986"
ALT_COLOR     = "#f0f4ff"
WHITE         = "#ffffff"

def fmt_p(p):
    # Format a p-value, collapsing very small ones to "< 0.001"
    if p != p: return "n/a"
    if p < 0.001: return "< 0.001"
    return f"{p:.4f}"

def fmt_val(v):
    # Format a number to four decimal places, returning "n/a" for NaN
    if v != v: return "n/a"
    return f"{v:.4f}"

def make_header_cell(ax, text, x, y, w, h, fontsize=10):
    # Draw a dark header cell with white text, used along the top row and left column
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", facecolor=HEADER_COLOR, edgecolor="white", linewidth=0.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold")

def make_cell(ax, text, x, y, w, h, bg=WHITE, color="black", fontsize=9, bold=False):
    # Draw a normal data cell with a configurable background and text colour
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", facecolor=bg, edgecolor="#cccccc", linewidth=0.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold" if bold else "normal")

def sig_color(s):
    # Pick a colour for a significance marker so it stands out in the table
    if s == "***": return "#16a34a"
    if s == "**":  return "#ca8a04"
    if s == "*":   return "#ea580c"
    if s == "ns":  return "#6b7280"
    return "#6b7280"


# Table A: Shapiro-Wilk
def draw_shapiro_table(data: dict, cat_order: list, filename: str):
    comparisons = ["normal_vs_misrouted", "normal_vs_top1", "misrouted_vs_top1"]
    comp_labels = ["Normal vs Misrouted", "Normal vs Top-1", "Misrouted vs Top-1"]
    col_labels  = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(comparisons)
    n_cols = len(col_labels)
    col_w, row_h, lbl_w = 2.0, 0.8, 2.0
    fig, ax = plt.subplots(figsize=(lbl_w + n_cols * col_w + 0.4, row_h * (n_rows + 1) + 1.4))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Shapiro-Wilk Test on Difference Scores — with Selected Test",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Comparison", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    col_keys = ["overall"] + cat_order
    all_data = {"overall": data["overall"], **{c: data[c] for c in cat_order}}

    for i, (comp, comp_label) in enumerate(zip(comparisons, comp_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = ALT_COLOR if i % 2 == 0 else WHITE
        make_cell(ax, comp_label, 0, y, lbl_w, row_h,
                  bg=SUBHEAD_COLOR, color="white", bold=True, fontsize=9)
        for j, col_key in enumerate(col_keys):
            sw         = all_data[col_key]["shapiro_wilk_differences"][comp]
            test       = all_data[col_key]["test_used"][comp]
            verdict    = sw["normal"]
            test_short = "t-test" if test == "paired t-test" else "Wilcoxon"
            cell_text  = f"W={fmt_val(sw['W'])}\np={fmt_p(sw['p'])}\nNormal: {verdict}\n→ {test_short}"
            color = "#16a34a" if verdict == "yes" else "#dc2626"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h,
                      bg=bg, color=color, fontsize=7.5)

    ax.text(0, -0.18,
            "Normal: p ≥ 0.05 → paired t-test  |  Not normal: p < 0.05 → Wilcoxon signed-rank",
            fontsize=7.5, color="#444444", transform=ax.transData)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# Table B: Main test results including effect sizes
def draw_test_results_table(data: dict, cat_order: list, filename: str):
    comparisons = ["normal_vs_misrouted", "normal_vs_top1", "misrouted_vs_top1"]
    comp_labels = ["Normal vs Misrouted", "Normal vs Top-1", "Misrouted vs Top-1"]
    col_labels  = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(comparisons)
    n_cols = len(col_labels)
    col_w, row_h, lbl_w = 2.0, 1.0, 2.0  # rows are a bit taller here to fit the effect size line in
    fig, ax = plt.subplots(figsize=(lbl_w + n_cols * col_w + 0.4, row_h * (n_rows + 1) + 1.6))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Statistical Test Results with Effect Sizes",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Comparison", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    col_keys = ["overall"] + cat_order
    all_data = {"overall": data["overall"], **{c: data[c] for c in cat_order}}

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
                es_str   = f"d = {result['effect_size']:.3f}" if result['effect_size'] == result['effect_size'] else "d = n/a"
            else:
                stat_str = f"W = {result['W']:.1f}" if result['W'] == result['W'] else "W = n/a"
                es_str   = f"r = {result['effect_size']:.3f}" if result['effect_size'] == result['effect_size'] else "r = n/a"
            p         = fmt_p(result["p"])
            s         = result["sig"]
            mag       = result.get("effect_label", "n/a")
            cell_text = f"{stat_str}\np = {p} {s}\n{es_str} ({mag})\n[{test_short}]"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h,
                      bg=bg, color=sig_color(s), fontsize=7.5, bold=(s not in ("ns", "n/a")))

    ax.text(0, -0.18,
            "*** p<0.001  ** p<0.01  * p<0.05  ns=not significant  |  "
            "Cohen's d: small=0.2, medium=0.5, large=0.8  |  r: small=0.1, medium=0.3, large=0.5",
            fontsize=7.5, color="#444444", transform=ax.transData)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# Table C: Descriptive stats
def draw_descriptive_table(data: dict, cat_order: list, filename: str):
    conditions  = ["normal", "misrouted", "top1"]
    cond_labels = ["Normal", "Misrouted", "Top-1"]
    col_labels  = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(conditions)
    n_cols = len(col_labels)
    col_w, row_h, lbl_w = 2.0, 0.7, 1.4
    fig, ax = plt.subplots(figsize=(lbl_w + n_cols * col_w + 0.4, row_h * (n_rows + 1) + 1.0))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Descriptive Statistics — Mean ± SD Semantic Entropy by Condition & Category",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Condition", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    col_keys = ["overall"] + cat_order
    all_data = {"overall": data["overall"], **{c: data[c] for c in cat_order}}
    colors_cond = {"normal": "#dbeafe", "misrouted": "#fef3c7", "top1": "#fee2e2"}

    for i, (cond, cond_label) in enumerate(zip(conditions, cond_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = colors_cond[cond]
        make_cell(ax, cond_label, 0, y, lbl_w, row_h,
                  bg=SUBHEAD_COLOR, color="white", bold=True)
        for j, col_key in enumerate(col_keys):
            mean = all_data[col_key]["means"][cond]
            std  = all_data[col_key]["stds"][cond]
            make_cell(ax, f"{mean:.4f}\n± {std:.4f}",
                      lbl_w + j * col_w, y, col_w, row_h, bg=bg, fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# Generate all tables
cat_order = ["unanswerable", "open_ended", "factual"]
combined  = {
    "overall":      overall_results,
    "unanswerable": category_results.get("unanswerable", {}),
    "open_ended":   category_results.get("open_ended",   {}),
    "factual":      category_results.get("factual",      {}),
}

draw_shapiro_table(combined,      cat_order, "results/table_shapiro_wilk.png")
draw_test_results_table(combined, cat_order, "results/table_test_results.png")
draw_descriptive_table(combined,  cat_order, "results/table_descriptive.png")

print("\nAll tables saved to results/")
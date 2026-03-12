import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
per_prompt_path = "results/per_prompt_comparison.json"


# ── Helpers ────────────────────────────────────────────────────────────────────
def sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_shapiro(arr: np.ndarray, label: str) -> dict:
    """Shapiro-Wilk normality test. Returns W statistic, p-value, and sig."""
    # Shapiro-Wilk requires at least 3 samples and all values not identical
    if len(arr) < 3 or np.all(arr == arr[0]):
        return {"W": float("nan"), "p": float("nan"), "sig": "n/a", "normal": "n/a"}
    w, p = stats.shapiro(arr)
    normal = "yes" if p >= 0.05 else "no"
    return {"W": float(w), "p": float(p), "sig": sig(p), "normal": normal}


def run_ttests(normal: np.ndarray, misrouted: np.ndarray, top1: np.ndarray, label: str) -> dict:
    t1, p1 = stats.ttest_rel(normal, misrouted)
    t2, p2 = stats.ttest_rel(normal, top1)
    t3, p3 = stats.ttest_rel(misrouted, top1)

    print(f"\n{'='*60}")
    print(f"PAIRED T-TEST RESULTS — {label}")
    print(f"{'='*60}")
    print(f"N: {len(normal)}\n")

    print(f"{'Condition':<12} {'Mean':>8} {'Std':>8}")
    print("-"*32)
    print(f"{'Normal':<12} {normal.mean():>8.4f} {normal.std():>8.4f}")
    print(f"{'Misrouted':<12} {misrouted.mean():>8.4f} {misrouted.std():>8.4f}")
    print(f"{'Top-1':<12} {top1.mean():>8.4f} {top1.std():>8.4f}")

    print(f"\n{'Comparison':<28} {'t':>8} {'p':>10} {'sig':>6}")
    print("-"*56)
    print(f"{'Normal vs Misrouted':<28} {t1:>8.4f} {p1:>10.4f} {sig(p1):>6}")
    print(f"{'Normal vs Top-1':<28} {t2:>8.4f} {p2:>10.4f} {sig(p2):>6}")
    print(f"{'Misrouted vs Top-1':<28} {t3:>8.4f} {p3:>10.4f} {sig(p3):>6}")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")

    return {
        "n": len(normal),
        "means": {"normal": float(normal.mean()), "misrouted": float(misrouted.mean()), "top1": float(top1.mean())},
        "stds":  {"normal": float(normal.std()),  "misrouted": float(misrouted.std()),  "top1": float(top1.std())},
        "paired_ttests": {
            "normal_vs_misrouted": {"t": float(t1), "p": float(p1), "sig": sig(p1)},
            "normal_vs_top1":      {"t": float(t2), "p": float(p2), "sig": sig(p2)},
            "misrouted_vs_top1":   {"t": float(t3), "p": float(p3), "sig": sig(p3)},
        }
    }


def run_shapiro_block(normal: np.ndarray, misrouted: np.ndarray, top1: np.ndarray, label: str) -> dict:
    sw_n = run_shapiro(normal,    "Normal")
    sw_m = run_shapiro(misrouted, "Misrouted")
    sw_t = run_shapiro(top1,      "Top-1")

    print(f"\n{'='*60}")
    print(f"SHAPIRO-WILK NORMALITY TEST — {label}")
    print(f"{'='*60}")
    print(f"{'Condition':<12} {'W':>8} {'p':>10} {'sig':>6} {'Normal?':>8}")
    print("-"*48)
    for name, sw in [("Normal", sw_n), ("Misrouted", sw_m), ("Top-1", sw_t)]:
        W = f"{sw['W']:.4f}" if sw['W'] == sw['W'] else "n/a"
        p = f"{sw['p']:.4f}" if sw['p'] == sw['p'] else "n/a"
        print(f"{name:<12} {W:>8} {p:>10} {sw['sig']:>6} {sw['normal']:>8}")
    print("Normal? = p >= 0.05 (fail to reject normality)")

    return {"normal": sw_n, "misrouted": sw_m, "top1": sw_t}


def run_wilcoxon(normal: np.ndarray, misrouted: np.ndarray, top1: np.ndarray, label: str) -> dict:
    """Wilcoxon signed-rank test — non-parametric alternative to paired t-test."""

    def wilcoxon_pair(a: np.ndarray, b: np.ndarray):
        diff = a - b
        if np.all(diff == 0):
            return float("nan"), float("nan")
        try:
            stat, p = stats.wilcoxon(a, b)
            return float(stat), float(p)
        except ValueError:
            return float("nan"), float("nan")

    s1, p1 = wilcoxon_pair(normal, misrouted)
    s2, p2 = wilcoxon_pair(normal, top1)
    s3, p3 = wilcoxon_pair(misrouted, top1)

    print(f"\n{'='*60}")
    print(f"WILCOXON SIGNED-RANK TEST — {label}")
    print(f"{'='*60}")
    print(f"(Non-parametric alternative to paired t-test)")
    print(f"\n{'Comparison':<28} {'W':>10} {'p':>10} {'sig':>6}")
    print("-"*56)
    for name, s, p in [("Normal vs Misrouted", s1, p1),
                        ("Normal vs Top-1",     s2, p2),
                        ("Misrouted vs Top-1",  s3, p3)]:
        W_str = f"{s:.1f}" if s == s else "n/a"
        p_str = f"{p:.4f}" if p == p else "n/a"
        s_str = sig(p) if p == p else "n/a"
        print(f"{name:<28} {W_str:>10} {p_str:>10} {s_str:>6}")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")

    return {
        "normal_vs_misrouted": {"W": s1, "p": p1, "sig": sig(p1) if p1 == p1 else "n/a"},
        "normal_vs_top1":      {"W": s2, "p": p2, "sig": sig(p2) if p2 == p2 else "n/a"},
        "misrouted_vs_top1":   {"W": s3, "p": p3, "sig": sig(p3) if p3 == p3 else "n/a"},
    }


# ── Load per-prompt entropy scores ─────────────────────────────────────────────
with open(per_prompt_path, "r") as f:
    prompt_rows = json.load(f)

normal_all    = np.array([r["normal_entropy"]    for r in prompt_rows])
misrouted_all = np.array([r["misrouted_entropy"] for r in prompt_rows])
top1_all      = np.array([r["top1_entropy"]      for r in prompt_rows])

# ── Overall tests ──────────────────────────────────────────────────────────────
overall_sw      = run_shapiro_block(normal_all, misrouted_all, top1_all, "OVERALL (n=81)")
overall_ttests  = run_ttests(normal_all, misrouted_all, top1_all, "OVERALL (n=81)")
overall_wc      = run_wilcoxon(normal_all, misrouted_all, top1_all, "OVERALL (n=81)")
overall_results = {**overall_ttests, "shapiro_wilk": overall_sw, "wilcoxon": overall_wc}

# ── Per-category tests ─────────────────────────────────────────────────────────
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
    sw    = run_shapiro_block(n_arr, r_arr, t_arr, f"{category.upper()} (n={len(n_arr)})")
    tt    = run_ttests(n_arr, r_arr, t_arr, f"{category.upper()} (n={len(n_arr)})")
    wc    = run_wilcoxon(n_arr, r_arr, t_arr, f"{category.upper()} (n={len(n_arr)})")
    category_results[category] = {**tt, "shapiro_wilk": sw, "wilcoxon": wc}

# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {"overall": overall_results, "by_category": category_results}
with open("results/ttest_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to results/ttest_results.json")


# ── Table visualisation helpers ────────────────────────────────────────────────
HEADER_COLOR  = "#1e3a5f"
SUBHEAD_COLOR = "#2d5986"
ALT_COLOR     = "#f0f4ff"
DIFF_COLOR    = "#fff7f7"
WHITE         = "#ffffff"

def fmt_p(p):
    if p != p: return "n/a"          # NaN check
    if p < 0.001: return "< 0.001"
    return f"{p:.4f}"

def fmt_val(v):
    if v != v: return "n/a"
    return f"{v:.4f}"

def make_header_cell(ax, text, x, y, w, h, fontsize=10):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", facecolor=HEADER_COLOR, edgecolor="white", linewidth=0.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold")

def make_cell(ax, text, x, y, w, h, bg=WHITE, color="black", fontsize=9, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="square,pad=0", facecolor=bg, edgecolor="#cccccc", linewidth=0.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold" if bold else "normal")

def sig_color(s):
    if s == "***": return "#16a34a"
    if s == "**":  return "#ca8a04"
    if s == "*":   return "#ea580c"
    if s == "ns":  return "#6b7280"
    return "#6b7280"


# ── Table A: Shapiro-Wilk ──────────────────────────────────────────────────────
def draw_shapiro_table(sw_data: dict, cat_order: list, filename: str):
    """
    Rows: conditions (Normal, Misrouted, Top-1)
    Cols: Overall + each category
    Shows W, p, and normality verdict per cell
    """
    conditions  = ["normal", "misrouted", "top1"]
    cond_labels = ["Normal", "Misrouted", "Top-1"]
    col_labels  = ["Overall (n=81)"] + [c.replace("_", " ").title() + " (n=27)" for c in cat_order]

    n_rows = len(conditions)
    n_cols = len(col_labels)

    col_w  = 1.8
    row_h  = 0.7
    lbl_w  = 1.4
    fig_w  = lbl_w + n_cols * col_w + 0.4
    fig_h  = row_h * (n_rows + 1) + 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, lbl_w + n_cols * col_w)
    ax.set_ylim(0, row_h * (n_rows + 1))
    ax.axis("off")
    fig.suptitle("Shapiro-Wilk Normality Test — Semantic Entropy by Condition & Category",
                 fontsize=12, fontweight="bold", y=0.97)

    # Corner cell
    make_header_cell(ax, "Condition", 0, n_rows * row_h, lbl_w, row_h)
    # Column headers
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    # Data rows
    all_sw = {"overall": sw_data["overall"]}
    for cat in cat_order:
        all_sw[cat] = sw_data[cat]
    col_keys = ["overall"] + cat_order

    for i, (cond, cond_label) in enumerate(zip(conditions, cond_labels)):
        y = (n_rows - 1 - i) * row_h
        bg = ALT_COLOR if i % 2 == 0 else WHITE
        # Row label
        make_cell(ax, cond_label, 0, y, lbl_w, row_h, bg=SUBHEAD_COLOR, color="white", bold=True)
        for j, col_key in enumerate(col_keys):
            sw  = all_sw[col_key][cond]
            W   = fmt_val(sw["W"])
            p   = fmt_p(sw["p"])
            s   = sw["sig"]
            verdict = sw["normal"]
            cell_text = f"W={W}\np={p}  {s}\nNormal: {verdict}"
            sc = sig_color(s) if s not in ("n/a", "ns") else ("black" if verdict == "yes" else "#dc2626")
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h, bg=bg, color=sc, fontsize=7.5)

    # Legend
    ax.text(0, -0.15, "Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant  |  Normal: p ≥ 0.05 = normally distributed",
            fontsize=7.5, color="#444444", transform=ax.transData)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# ── Table B: Paired t-tests ────────────────────────────────────────────────────
def draw_ttest_table(tt_data: dict, cat_order: list, filename: str):
    """
    Rows: each pairwise comparison
    Cols: Overall + each category
    Shows t, p, sig per cell
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
    fig.suptitle("Paired t-Test Results — Semantic Entropy by Comparison & Category",
                 fontsize=12, fontweight="bold", y=0.97)

    # Corner + column headers
    make_header_cell(ax, "Comparison", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    all_tt = {"overall": tt_data["overall"]}
    for cat in cat_order:
        all_tt[cat] = tt_data[cat]
    col_keys = ["overall"] + cat_order

    for i, (comp, comp_label) in enumerate(zip(comparisons, comp_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = ALT_COLOR if i % 2 == 0 else WHITE
        make_cell(ax, comp_label, 0, y, lbl_w, row_h, bg=SUBHEAD_COLOR, color="white", bold=True, fontsize=9)
        for j, col_key in enumerate(col_keys):
            tt   = all_tt[col_key]["paired_ttests"][comp]
            t    = f"{tt['t']:.3f}" if tt['t'] == tt['t'] else "n/a"
            p    = fmt_p(tt["p"])
            s    = tt["sig"]
            cell_text = f"t = {t}\np = {p}\n{s}"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h,
                      bg=bg, color=sig_color(s), fontsize=8, bold=(s != "ns"))

    # Means sub-table below
    y_offset = -row_h * 0.3
    ax.text(lbl_w / 2, y_offset, "Condition means:",
            ha="center", va="top", fontsize=8, fontweight="bold", color="#333333")

    ax.text(0, -0.18,
            "Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
            fontsize=7.5, color="#444444", transform=ax.transData)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# ── Table C: Descriptive stats ─────────────────────────────────────────────────
def draw_descriptive_table(tt_data: dict, cat_order: list, filename: str):
    """Mean ± SD for each condition × category."""
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

    all_tt = {"overall": tt_data["overall"]}
    for cat in cat_order:
        all_tt[cat] = tt_data[cat]
    col_keys = ["overall"] + cat_order

    colors_cond = {"normal": "#dbeafe", "misrouted": "#fef3c7", "top1": "#fee2e2"}

    for i, (cond, cond_label) in enumerate(zip(conditions, cond_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = colors_cond[cond]
        make_cell(ax, cond_label, 0, y, lbl_w, row_h, bg=SUBHEAD_COLOR, color="white", bold=True)
        for j, col_key in enumerate(col_keys):
            mean = all_tt[col_key]["means"][cond]
            std  = all_tt[col_key]["stds"][cond]
            cell_text = f"{mean:.4f}\n± {std:.4f}"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h, bg=bg, fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# ── Generate all tables ────────────────────────────────────────────────────────
cat_order = ["unanswerable", "open_ended", "factual"]

# Build combined data dict for table functions
combined = {
    "overall":      {**overall_results},
    "unanswerable": {**category_results.get("unanswerable", {})},
    "open_ended":   {**category_results.get("open_ended", {})},
    "factual":      {**category_results.get("factual", {})},
}

# Shapiro-Wilk table needs sw nested under each key
sw_combined = {
    "overall":      overall_results["shapiro_wilk"],
    "unanswerable": category_results.get("unanswerable", {}).get("shapiro_wilk", {}),
    "open_ended":   category_results.get("open_ended",   {}).get("shapiro_wilk", {}),
    "factual":      category_results.get("factual",      {}).get("shapiro_wilk", {}),
}

draw_shapiro_table(sw_combined,  cat_order, "results/table_shapiro_wilk.png")
draw_ttest_table(combined,       cat_order, "results/table_ttests.png")
draw_descriptive_table(combined, cat_order, "results/table_descriptive.png")


# ── Table D: Wilcoxon signed-rank ─────────────────────────────────────────────
def draw_wilcoxon_table(tt_data: dict, cat_order: list, filename: str):
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
    fig.suptitle("Wilcoxon Signed-Rank Test — Semantic Entropy by Comparison & Category\n"
                 "(Non-parametric alternative to paired t-test)",
                 fontsize=12, fontweight="bold", y=0.97)

    make_header_cell(ax, "Comparison", 0, n_rows * row_h, lbl_w, row_h)
    for j, cl in enumerate(col_labels):
        make_header_cell(ax, cl, lbl_w + j * col_w, n_rows * row_h, col_w, row_h, fontsize=9)

    all_wc = {"overall": tt_data["overall"]}
    for cat in cat_order:
        all_wc[cat] = tt_data[cat]
    col_keys = ["overall"] + cat_order

    for i, (comp, comp_label) in enumerate(zip(comparisons, comp_labels)):
        y  = (n_rows - 1 - i) * row_h
        bg = ALT_COLOR if i % 2 == 0 else WHITE
        make_cell(ax, comp_label, 0, y, lbl_w, row_h, bg=SUBHEAD_COLOR, color="white", bold=True, fontsize=9)
        for j, col_key in enumerate(col_keys):
            wc   = all_wc[col_key]["wilcoxon"][comp]
            W    = f"{wc['W']:.1f}" if wc['W'] == wc['W'] else "n/a"
            p    = fmt_p(wc["p"])
            s    = wc["sig"]
            cell_text = f"W = {W}\np = {p}\n{s}"
            make_cell(ax, cell_text, lbl_w + j * col_w, y, col_w, row_h,
                      bg=bg, color=sig_color(s), fontsize=8, bold=(s not in ("ns", "n/a")))

    ax.text(0, -0.18,
            "Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant  |  W = Wilcoxon test statistic",
            fontsize=7.5, color="#444444", transform=ax.transData)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


draw_wilcoxon_table(combined, cat_order, "results/table_wilcoxon.png")

print("\nAll tables saved to results/")

import json
import numpy as np
from scipy import stats
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
# Reads from the output of semantic_entropy.py — run that first
per_prompt_path = "results/per_prompt_comparison.json"


# ── Helpers ────────────────────────────────────────────────────────────────────
def sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_ttests(normal: np.ndarray, random: np.ndarray, bottom: np.ndarray, label: str):
    t1, p1 = stats.ttest_rel(normal, random)
    t2, p2 = stats.ttest_rel(normal, bottom)
    t3, p3 = stats.ttest_rel(random, bottom)

    print(f"\n{'='*60}")
    print(f"PAIRED T-TEST RESULTS — {label}")
    print(f"{'='*60}")
    print(f"N: {len(normal)}\n")

    print(f"{'Condition':<12} {'Mean':>8} {'Std':>8}")
    print("-"*32)
    print(f"{'Normal':<12} {normal.mean():>8.4f} {normal.std():>8.4f}")
    print(f"{'Random':<12} {random.mean():>8.4f} {random.std():>8.4f}")
    print(f"{'Bottom-4':<12} {bottom.mean():>8.4f} {bottom.std():>8.4f}")

    print(f"\n{'Comparison':<28} {'t':>8} {'p':>10} {'sig':>6}")
    print("-"*56)
    print(f"{'Normal vs Random':<28} {t1:>8.4f} {p1:>10.4f} {sig(p1):>6}")
    print(f"{'Normal vs Bottom-4':<28} {t2:>8.4f} {p2:>10.4f} {sig(p2):>6}")
    print(f"{'Random vs Bottom-4':<28} {t3:>8.4f} {p3:>10.4f} {sig(p3):>6}")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")

    return {
        "n": len(normal),
        "means":  {"normal": float(normal.mean()), "random": float(random.mean()), "bottom_4": float(bottom.mean())},
        "stds":   {"normal": float(normal.std()),  "random": float(random.std()),  "bottom_4": float(bottom.std())},
        "paired_ttests": {
            "normal_vs_random":  {"t": float(t1), "p": float(p1), "sig": sig(p1)},
            "normal_vs_bottom4": {"t": float(t2), "p": float(p2), "sig": sig(p2)},
            "random_vs_bottom4": {"t": float(t3), "p": float(p3), "sig": sig(p3)},
        }
    }


# ── Load per-prompt entropy scores ─────────────────────────────────────────────
with open(per_prompt_path, "r") as f:
    prompt_rows = json.load(f)

normal_all = np.array([r["normal_entropy"] for r in prompt_rows])
random_all = np.array([r["random_entropy"] for r in prompt_rows])
bottom_all = np.array([r["bottom_entropy"] for r in prompt_rows])

# ── Overall t-tests (n=81) ─────────────────────────────────────────────────────
overall_results = run_ttests(normal_all, random_all, bottom_all, "OVERALL (n=81)")

# ── Per-category t-tests (n=27 each) ──────────────────────────────────────────
cat_data = defaultdict(lambda: {"normal": [], "random": [], "bottom": []})
for row in prompt_rows:
    cat_data[row["category"]]["normal"].append(row["normal_entropy"])
    cat_data[row["category"]]["random"].append(row["random_entropy"])
    cat_data[row["category"]]["bottom"].append(row["bottom_entropy"])

category_results = {}
for category, vals in cat_data.items():
    n_arr = np.array(vals["normal"])
    r_arr = np.array(vals["random"])
    b_arr = np.array(vals["bottom"])
    category_results[category] = run_ttests(n_arr, r_arr, b_arr, f"{category.upper()} (n={len(n_arr)})")

# ── Save all results ───────────────────────────────────────────────────────────
output = {
    "overall":     overall_results,
    "by_category": category_results,
}

with open("results/ttest_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to results/ttest_results.json")
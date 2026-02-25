import json
import csv
import re
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
NORMAL_PATH    = "results/normal.jsonl"
MISROUTED_PATH = "results/misrouted.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Adaptive thresholds based on mean response length (in words) for the batch.
# Short responses (< 20 words): need lower threshold — "Paris." and
#   "The capital of France is Paris." are clearly equivalent but score ~0.65.
# Long responses (> 80 words): need higher threshold — two different paragraphs
#   about climate change will score 0.75+ even if making different points.
# Medium falls in between.
SHORT_THRESHOLD  = 0.60   # for mean response length < 20 words
MEDIUM_THRESHOLD = 0.72   # for mean response length 20-80 words
LONG_THRESHOLD   = 0.82   # for mean response length > 80 words


def get_threshold(responses: list[str]) -> float:
    mean_words = np.mean([len(r.split()) for r in responses])
    if mean_words < 20:
        return SHORT_THRESHOLD
    elif mean_words < 80:
        return MEDIUM_THRESHOLD
    else:
        return LONG_THRESHOLD


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Using device: {DEVICE}")
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def cluster_responses(responses: list[str], embeddings: torch.Tensor) -> list[list[int]]:
    """
    Greedy centroid-based clustering with adaptive threshold.

    Uses the mean response length to pick an appropriate similarity threshold.
    Each cluster tracks its centroid (running mean of member embeddings).
    A new response joins the first cluster whose centroid exceeds the threshold.
    """
    threshold = get_threshold(responses)
    clusters: list[list[int]] = []
    centroids: list[torch.Tensor] = []

    for i in range(len(responses)):
        emb = embeddings[i]
        placed = False

        for j, centroid in enumerate(centroids):
            sim = F.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0)).item()
            if sim >= threshold:
                clusters[j].append(i)
                n = len(clusters[j])
                centroids[j] = F.normalize((centroid * (n - 1) + emb) / n, dim=0)
                placed = True
                break

        if not placed:
            clusters.append([i])
            centroids.append(F.normalize(emb.clone(), dim=0))

    return clusters


def semantic_entropy(clusters: list[list[int]]) -> float:
    sizes = np.array([len(c) for c in clusters], dtype=float)
    if len(sizes) == 1:
        return 0.0
    p = sizes / sizes.sum()
    return float(-(p * np.log(p)).sum())


# ── Core processing ────────────────────────────────────────────────────────────
def process(items: list[dict], label: str) -> list[dict]:
    out = []
    for i, item in enumerate(items):
        responses = item["responses"]
        mean_words = np.mean([len(r.split()) for r in responses])
        threshold  = get_threshold(responses)
        print(f"  [{label}] {i+1}/{len(items)}: {item['prompt'][:55]}... "
              f"(avg {mean_words:.0f} words, threshold={threshold})")

        embeddings = embed_model.encode(
            responses,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE,
            show_progress_bar=False,
        )

        clusters = cluster_responses(responses, embeddings)
        entropy  = semantic_entropy(clusters)

        out.append({
            "prompt":        item["prompt"],
            "category":      item["category"],
            "num_responses": len(responses),
            "num_clusters":  len(clusters),
            "entropy":       entropy,
        })
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\nLoading files...")
    normal_items = load_jsonl(NORMAL_PATH)
    mis_items    = load_jsonl(MISROUTED_PATH)

    if len(normal_items) != len(mis_items):
        raise ValueError(
            f"Line count mismatch: normal={len(normal_items)} misrouted={len(mis_items)}"
        )
    for i, (n, m) in enumerate(zip(normal_items, mis_items)):
        if n.get("prompt") != m.get("prompt"):
            raise ValueError(f"Prompt mismatch at line {i}")

    print("\nComputing semantic entropy (normal)...")
    normal_res = process(normal_items, "normal")

    print("\nComputing semantic entropy (misrouted)...")
    mis_res = process(mis_items, "misrouted")

    # ── Per-prompt comparison ──────────────────────────────────────────────────
    prompt_rows = []
    for n, m in zip(normal_res, mis_res):
        prompt_rows.append({
            "prompt":             n["prompt"],
            "category":           n["category"],
            "normal_clusters":    n["num_clusters"],
            "normal_entropy":     round(n["entropy"], 4),
            "misrouted_clusters": m["num_clusters"],
            "misrouted_entropy":  round(m["entropy"], 4),
            "entropy_diff":       round(m["entropy"] - n["entropy"], 4),
        })

    with open("results/per_prompt_comparison.json", "w", encoding="utf-8") as f:
        json.dump(prompt_rows, f, indent=2)

    with open("results/per_prompt_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prompt_rows)

    # ── Per-category comparison ────────────────────────────────────────────────
    cat = defaultdict(lambda: {"normal": [], "misrouted": []})
    for n, m in zip(normal_res, mis_res):
        cat[n["category"]]["normal"].append(n["entropy"])
        cat[n["category"]]["misrouted"].append(m["entropy"])

    category_rows = []
    for category, vals in cat.items():
        n_vals = np.array(vals["normal"],    dtype=float)
        m_vals = np.array(vals["misrouted"], dtype=float)
        category_rows.append({
            "category":               category,
            "num_prompts":            int(len(n_vals)),
            "normal_mean_entropy":    float(n_vals.mean()),
            "normal_std_entropy":     float(n_vals.std()),
            "misrouted_mean_entropy": float(m_vals.mean()),
            "misrouted_std_entropy":  float(m_vals.std()),
            "mean_entropy_diff":      float(m_vals.mean() - n_vals.mean()),
        })

    with open("results/per_category_comparison.json", "w", encoding="utf-8") as f:
        json.dump(category_rows, f, indent=2)

    with open("results/per_category_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== CATEGORY SUMMARY ===")
    for row in sorted(category_rows, key=lambda r: r["category"]):
        print(f"\n{row['category']} ({row['num_prompts']} prompts)")
        print(f"  Normal    - mean entropy: {row['normal_mean_entropy']:.4f}  (std: {row['normal_std_entropy']:.4f})")
        print(f"  Misrouted - mean entropy: {row['misrouted_mean_entropy']:.4f}  (std: {row['misrouted_std_entropy']:.4f})")
        print(f"  Difference: {row['mean_entropy_diff']:+.4f}")

    print("\nSaved:")
    print("  results/per_prompt_comparison.json")
    print("  results/per_prompt_comparison.csv")
    print("  results/per_category_comparison.json")
    print("  results/per_category_comparison.csv")


if __name__ == "__main__":
    main()
    
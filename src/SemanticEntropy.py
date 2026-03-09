import json, csv, torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

normal_path  = "results/normal.jsonl"
random_path  = "results/misrouted.jsonl"
bottom_path  = "results/misrouted_bottom.jsonl"

embed_model_name = "all-mpnet-base-v2"

short_threshold  = 0.60
medium_threshold = 0.72
long_threshold   = 0.82

# Load the embedding model - converts sentences into lists of numbers
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(embed_model_name, device=device)


# Short responses need a lower threshold
def get_threshold(responses: list[str]) -> float:
    mean_words = np.mean([len(r.split()) for r in responses])
    if mean_words < 20:
        return short_threshold
    elif mean_words < 80:
        return medium_threshold
    else:
        return long_threshold


# Group the responses into clusters that say the same thing
def cluster_responses(responses: list[str], embeddings: torch.Tensor) -> list[list[int]]:
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


# Produces a number representing how varied the responses were
def semantic_entropy(clusters: list[list[int]]) -> float:
    sizes = np.array([len(c) for c in clusters], dtype=float)
    if len(sizes) == 1:
        return 0.0
    p = sizes / sizes.sum()
    return float(-(p * np.log(p)).sum())


# Reads jsonl file containing the responses and returns it as a list
def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_entropy_for_items(items: list[dict], label: str) -> list[dict]:
    """Compute semantic entropy for every prompt in a condition."""
    results = []
    for i, item in enumerate(items):
        responses  = item["responses"]
        mean_words = np.mean([len(r.split()) for r in responses])
        threshold  = get_threshold(responses)
        print(f"  [{label}] {i+1}/{len(items)}: {item['prompt'][:55]}... "
              f"(avg {mean_words:.0f} words, threshold={threshold})")

        embeddings = embed_model.encode(
            responses,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device,
            show_progress_bar=False,
        )

        clusters = cluster_responses(responses, embeddings)
        entropy  = semantic_entropy(clusters)

        results.append({
            "prompt":        item["prompt"],
            "category":      item["category"],
            "num_responses": len(responses),
            "num_clusters":  len(clusters),
            "entropy":       entropy,
        })
    return results


# Load all 3 conditions
normal_items = load_jsonl(normal_path)
random_items = load_jsonl(random_path)
bottom_items = load_jsonl(bottom_path)

# Verify all 3 files match up
if not (len(normal_items) == len(random_items) == len(bottom_items)):
    raise ValueError(f"Line count mismatch: normal={len(normal_items)} "
                     f"random={len(random_items)} bottom={len(bottom_items)}")

for i, (n, r, b) in enumerate(zip(normal_items, random_items, bottom_items)):
    if not (n.get("prompt") == r.get("prompt") == b.get("prompt")):
        raise ValueError(f"Prompt mismatch at line {i}")

# Compute entropy for each condition
print("Computing entropy for normal routing...")
normal_res = compute_entropy_for_items(normal_items, "normal")

print("Computing entropy for random routing...")
random_res = compute_entropy_for_items(random_items, "random")

print("Computing entropy for bottom-4 routing...")
bottom_res = compute_entropy_for_items(bottom_items, "bottom4")

# ── Per-prompt comparison ──────────────────────────────────────────────────────
prompt_rows = []
for n, r, b in zip(normal_res, random_res, bottom_res):
    prompt_rows.append({
        "prompt":                n["prompt"],
        "category":              n["category"],
        "normal_clusters":       n["num_clusters"],
        "normal_entropy":        round(n["entropy"], 4),
        "random_clusters":       r["num_clusters"],
        "random_entropy":        round(r["entropy"], 4),
        "bottom_clusters":       b["num_clusters"],
        "bottom_entropy":        round(b["entropy"], 4),
        "diff_normal_random":    round(r["entropy"] - n["entropy"], 4),
        "diff_normal_bottom":    round(b["entropy"] - n["entropy"], 4),
        "diff_random_bottom":    round(b["entropy"] - r["entropy"], 4),
    })

with open("results/per_prompt_comparison.json", "w", encoding="utf-8") as f:
    json.dump(prompt_rows, f, indent=2)

with open("results/per_prompt_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0].keys()))
    writer.writeheader()
    writer.writerows(prompt_rows)

# ── Per-category summary ───────────────────────────────────────────────────────
cat = defaultdict(lambda: {"normal": [], "random": [], "bottom": []})
for n, r, b in zip(normal_res, random_res, bottom_res):
    cat[n["category"]]["normal"].append(n["entropy"])
    cat[n["category"]]["random"].append(r["entropy"])
    cat[n["category"]]["bottom"].append(b["entropy"])

category_rows = []
for category, vals in cat.items():
    n_vals = np.array(vals["normal"], dtype=float)
    r_vals = np.array(vals["random"], dtype=float)
    b_vals = np.array(vals["bottom"], dtype=float)
    category_rows.append({
        "category":              category,
        "num_prompts":           int(len(n_vals)),
        "normal_mean_entropy":   float(n_vals.mean()),
        "normal_std_entropy":    float(n_vals.std()),
        "random_mean_entropy":   float(r_vals.mean()),
        "random_std_entropy":    float(r_vals.std()),
        "bottom_mean_entropy":   float(b_vals.mean()),
        "bottom_std_entropy":    float(b_vals.std()),
        "diff_normal_random":    float(r_vals.mean() - n_vals.mean()),
        "diff_normal_bottom":    float(b_vals.mean() - n_vals.mean()),
        "diff_random_bottom":    float(b_vals.mean() - r_vals.mean()),
    })

with open("results/per_category_comparison.json", "w", encoding="utf-8") as f:
    json.dump(category_rows, f, indent=2)

with open("results/per_category_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
    writer.writeheader()
    writer.writerows(category_rows)

print("\nDone. Results saved to results/per_prompt_comparison.{json,csv} "
      "and results/per_category_comparison.{json,csv}")
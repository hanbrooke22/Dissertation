import json, csv, torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

normal_path  = "results/normal.jsonl"
random_path  = "results/misrouted.jsonl"
top1_path    = "results/top1.jsonl"

embed_model_name = "all-mpnet-base-v2"

# Different cut-offs for how alike two answers need to be to count as "the same idea", short answers get a looser cut-off so we don't end up splitting them too much
short_threshold  = 0.60
medium_threshold = 0.72
long_threshold   = 0.82

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(embed_model_name, device=device)

def get_threshold(responses: list[str]) -> float:
    # Pick which cut-off to use based on how long the answers are on average
    mean_words = np.mean([len(r.split()) for r in responses])
    if mean_words < 20:
        return short_threshold
    elif mean_words < 80:
        return medium_threshold
    else:
        return long_threshold


def cluster_responses(responses: list[str], embeddings: torch.Tensor) -> list[list[int]]:
    # Go through the answers one by one and drop each into the first cluster it's similar enough to, or start a new cluster if nothing fits
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
                # Update the clusters's "average" answer now that a new one has joined
                centroids[j] = F.normalize((centroid * (n - 1) + emb) / n, dim=0)
                placed = True
                break

        if not placed:
            # Nothing matched, so this answer creates a new cluster
            clusters.append([i])
            centroids.append(F.normalize(emb.clone(), dim=0))

    return clusters

def semantic_entropy(clusters: list[list[int]]) -> float:
    # Work out a single number that says how spread out the answers are across clusters, higher means more disagreement
    sizes = np.array([len(c) for c in clusters], dtype=float)
    if len(sizes) == 1:
        return 0.0
    p = sizes / sizes.sum()
    return float(-(p * np.log(p)).sum())

def load_jsonl(path: str) -> list[dict]:
    # Read a file where each line is its own little JSON record and put them all in a list
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

        # Turn each answer into a list of numbers that captures its meaning, so we can compare them properly
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


# Open up all three sets of results so we can compare them
normal_items = load_jsonl(normal_path)
random_items = load_jsonl(random_path)
top1_items   = load_jsonl(top1_path)

# Run the entropy calculation on each set in turn
normal_res = compute_entropy_for_items(normal_items, "normal")

random_res = compute_entropy_for_items(random_items, "misrouted")

top1_res = compute_entropy_for_items(top1_items, "top1")

# Build a row per prompt showing the scores from each condition side by side, plus the gaps between them
prompt_rows = []
for n, r, t in zip(normal_res, random_res, top1_res):
    prompt_rows.append({
        "prompt":                 n["prompt"],
        "category":               n["category"],
        "normal_clusters":        n["num_clusters"],
        "normal_entropy":         round(n["entropy"], 4),
        "misrouted_clusters":     r["num_clusters"],
        "misrouted_entropy":      round(r["entropy"], 4),
        "top1_clusters":          t["num_clusters"],
        "top1_entropy":           round(t["entropy"], 4),
        "diff_normal_misrouted":  round(r["entropy"] - n["entropy"], 4),
        "diff_normal_top1":       round(t["entropy"] - n["entropy"], 4),
        "diff_misrouted_top1":    round(t["entropy"] - r["entropy"], 4),
    })

# Save the per prompt table as JSON for easy reading and as CSV for opening in a spreadsheet
with open("results/per_prompt_comparison.json", "w", encoding="utf-8") as f:
    json.dump(prompt_rows, f, indent=2)

with open("results/per_prompt_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0].keys()))
    writer.writeheader()
    writer.writerows(prompt_rows)

# Now do the same thing but grouped by category, so we can see the average score for each topic and how much it bounces around
cat = defaultdict(lambda: {"normal": [], "misrouted": [], "top1": []})
for n, r, t in zip(normal_res, random_res, top1_res):
    cat[n["category"]]["normal"].append(n["entropy"])
    cat[n["category"]]["misrouted"].append(r["entropy"])
    cat[n["category"]]["top1"].append(t["entropy"])

category_rows = []
for category, vals in cat.items():
    n_vals = np.array(vals["normal"],    dtype=float)
    r_vals = np.array(vals["misrouted"], dtype=float)
    t_vals = np.array(vals["top1"],      dtype=float)
    category_rows.append({
        "category":                category,
        "num_prompts":             int(len(n_vals)),
        "normal_mean_entropy":     float(n_vals.mean()),
        "normal_std_entropy":      float(n_vals.std()),
        "misrouted_mean_entropy":  float(r_vals.mean()),
        "misrouted_std_entropy":   float(r_vals.std()),
        "top1_mean_entropy":       float(t_vals.mean()),
        "top1_std_entropy":        float(t_vals.std()),
        "diff_normal_misrouted":   float(r_vals.mean() - n_vals.mean()),
        "diff_normal_top1":        float(t_vals.mean() - n_vals.mean()),
        "diff_misrouted_top1":     float(t_vals.mean() - r_vals.mean()),
    })

# Save the per category summary in both formats too
with open("results/per_category_comparison.json", "w", encoding="utf-8") as f:
    json.dump(category_rows, f, indent=2)

with open("results/per_category_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
    writer.writeheader()
    writer.writerows(category_rows)
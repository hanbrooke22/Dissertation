import json, csv, torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

normal_path  = "results/normal.jsonl"
random_path  = "results/misrouted.jsonl"
top1_path    = "results/top1.jsonl"

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
top1_items   = load_jsonl(top1_path)

# Verify all 3 files match up
if not (len(normal_items) == len(random_items) == len(top1_items)):
    raise ValueError(f"Line count mismatch: normal={len(normal_items)} "
                     f"misrouted={len(random_items)} top1={len(top1_items)}")

for i, (n, r, t) in enumerate(zip(normal_items, random_items, top1_items)):
    if not (n.get("prompt") == r.get("prompt") == t.get("prompt")):
        raise ValueError(f"Prompt mismatch at line {i}")

# Compute entropy for each condition
print("Computing entropy for normal routing...")
normal_res = compute_entropy_for_items(normal_items, "normal")

print("Computing entropy for misrouted routing...")
random_res = compute_entropy_for_items(random_items, "misrouted")

print("Computing entropy for top-1 routing...")
top1_res   = compute_entropy_for_items(top1_items, "top1")

# ── Per-prompt comparison ──────────────────────────────────────────────────────
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

with open("results/per_prompt_comparison.json", "w", encoding="utf-8") as f:
    json.dump(prompt_rows, f, indent=2)

with open("results/per_prompt_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0].keys()))
    writer.writeheader()
    writer.writerows(prompt_rows)

# ── Per-category summary ───────────────────────────────────────────────────────
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

with open("results/per_category_comparison.json", "w", encoding="utf-8") as f:
    json.dump(category_rows, f, indent=2)

with open("results/per_category_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
    writer.writeheader()
    writer.writerows(category_rows)

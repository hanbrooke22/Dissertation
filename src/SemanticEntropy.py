import json
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nli_model_name = "cross-encoder/nli-deberta-v3-large"
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model.eval()


def check_entailment(premise, hypothesis, question="", strict=False):
    if question:
        premise = question + " " + premise
        hypothesis = question + " " + hypothesis

    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    predicted_class = logits.argmax().item()

    if strict:
        return predicted_class == 1
    else:
        return predicted_class in [1, 2]


def cluster_responses(responses, question="", strict=False):
    clusters = []

    for response in responses:
        placed = False
        for cluster in clusters:
            if (check_entailment(response, cluster[0], question, strict) and
                    check_entailment(cluster[0], response, question, strict)):
                cluster.append(response)
                placed = True
                break
        if not placed:
            clusters.append([response])

    return clusters


def clusters_to_ids(clusters):
    ids = []
    for cluster_id, cluster in enumerate(clusters):
        for _ in cluster:
            ids.append(cluster_id)
    return ids


def cluster_entropy(ids):
    n_generations = len(ids)
    counts = np.bincount(ids)
    probs = counts / n_generations
    return (-probs * np.log(probs)).sum()


def process_file(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            category = item["category"]
            responses = item["responses"]

            strict = category == "unanswerable"
            clusters = cluster_responses(responses, question=prompt, strict=strict)
            ids = clusters_to_ids(clusters)
            entropy = cluster_entropy(ids)

            results.append({
                "prompt": prompt,
                "category": category,
                "num_responses": len(responses),
                "num_clusters": len(clusters),
                "entropy": round(entropy, 4)
            })
    return results


print("Processing normal routing...")
normal_results = process_file("results/normal.jsonl")

print("Processing misrouted routing...")
misrouted_results = process_file("results/misrouted.jsonl")

# --- File 1: Per-prompt comparison ---
prompt_rows = []
for n, m in zip(normal_results, misrouted_results):
    prompt_rows.append({
        "prompt": n["prompt"],
        "category": n["category"],
        "normal_clusters": n["num_clusters"],
        "normal_entropy": n["entropy"],
        "misrouted_clusters": m["num_clusters"],
        "misrouted_entropy": m["entropy"],
        "entropy_diff": round(m["entropy"] - n["entropy"], 4)
    })

with open("results/per_prompt_comparison.json", "w") as f:
    json.dump(prompt_rows, f, indent=2)

print("\nPer-prompt comparison saved to results/per_prompt_comparison.json")

# --- File 2: Per-category comparison ---
category_stats = defaultdict(lambda: {
    "normal_entropies": [],
    "misrouted_entropies": []
})

for n, m in zip(normal_results, misrouted_results):
    cat = n["category"]
    category_stats[cat]["normal_entropies"].append(n["entropy"])
    category_stats[cat]["misrouted_entropies"].append(m["entropy"])

category_rows = []
for category, stats in category_stats.items():
    n_entropies = stats["normal_entropies"]
    m_entropies = stats["misrouted_entropies"]
    category_rows.append({
        "category": category,
        "num_prompts": len(n_entropies),
        "normal_mean_entropy": round(np.mean(n_entropies), 4),
        "normal_std_entropy": round(np.std(n_entropies), 4),
        "misrouted_mean_entropy": round(np.mean(m_entropies), 4),
        "misrouted_std_entropy": round(np.std(m_entropies), 4),
        "mean_entropy_diff": round(np.mean(m_entropies) - np.mean(n_entropies), 4)
    })

with open("results/per_category_comparison.json", "w") as f:
    json.dump(category_rows, f, indent=2)

print("Per-category comparison saved to results/per_category_comparison.json")

# Print category summary to terminal
print("\n=== CATEGORY SUMMARY ===")
for row in category_rows:
    print(f"\n{row['category']} ({row['num_prompts']} prompts)")
    print(f"  Normal    - mean entropy: {row['normal_mean_entropy']:.4f} (std: {row['normal_std_entropy']:.4f})")
    print(f"  Misrouted - mean entropy: {row['misrouted_mean_entropy']:.4f} (std: {row['misrouted_std_entropy']:.4f})")
    print(f"  Difference: {row['mean_entropy_diff']:+.4f}")
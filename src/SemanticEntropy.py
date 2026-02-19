import json
import torch
import numpy as np
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
        return predicted_class == 1  # entailment only
    else:
        return predicted_class in [1, 2]  # entailment or neutral


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


with open("results/normal.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        prompt = item["prompt"]
        category = item["category"]
        responses = item["responses"]

        # Use strict entailment for unanswerable, lenient for others
        strict = category == "unanswerable"

        clusters = cluster_responses(responses, question=prompt, strict=strict)
        ids = clusters_to_ids(clusters)
        entropy = cluster_entropy(ids)

        print(f"\nPrompt: {prompt}")
        print(f"Category: {category}")
        print(f"Clusters: {len(clusters)}, Entropy: {entropy:.4f}")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {len(cluster)} response(s)")

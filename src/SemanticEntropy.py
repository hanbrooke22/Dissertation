import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nli_model_name = "cross-encoder/nli-deberta-v3-large"

nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model.eval()

def check_entailment(premise, hypothesis):
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
    return predicted_class in [1,2]

def cluster_responses(responses):
    clusters = []

    for response in responses: 
        placed = False
        for cluster in clusters: 
            if check_entailment(response, cluster[0]) and check_entailment(cluster[0], response):
                cluster.append(response)
                placed = True 
                break
        if not placed: 
            clusters.append([response])
    return clusters

with open("results/normal.jsonl", "r") as f: 
    item = json.loads(f.readline())

print(f"Prompt: {item['prompt']}")
print(f"Number of responses: {len(item['responses'])}")
print()

clusters = cluster_responses(item["responses"])

print(f"Number of clusters: {len(clusters)}")
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1} ({len(cluster)} response(s)):")
    for r in cluster:
        print(f"  - {r}...")
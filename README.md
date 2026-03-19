# MoE Routing and Model Uncertainty

An empirical investigation into how expert routing strategies in Mixture-of-Experts (MoE) language models affect output uncertainty, measured via semantic entropy.

## Overview

This project examines whether manipulating the expert routing mechanism of a MoE language model (Qwen1.5-MoE-A2.7B-Chat) causes measurable changes in response consistency. Three routing conditions are compared:

- **Normal** — default routing behaviour (baseline)
- **Top-1** — routing restricted to the single highest-scoring expert per token
- **Misrouted** — routing logits corrupted with 40% Gaussian noise

For each condition, 10 responses are generated per prompt and semantic entropy is computed over the resulting response clusters to quantify uncertainty.

## Repository Structure

```
src/
├── model.py              # Loads Qwen1.5-MoE with 4-bit quantisation (QLoRA, NF4)
├── normal.py             # Baseline inference — standard routing
├── top1.py               # Top-1 routing condition — patches MoE router modules
├── wrong.py              # Misrouted condition — injects noise into gate layers
├── SemanticEntropy.py    # Response embedding, clustering, and entropy computation
├── statTest.py           # Statistical tests (Shapiro-Wilk, paired t-test, Wilcoxon)
├── visualisations.py     # All plots and result tables

data/
└── prompts.jsonl         # Input prompts (one JSON object per line)

results/                  # All outputs saved here (see below)
```

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is required to run the model. The model is loaded in 4-bit quantisation to reduce VRAM requirements. This project was developed on NVIDIA RTX 4070 GPUs running Linux — behaviour on other hardware may vary.

## How to Run

Scripts must be run in the following order as each stage depends on the previous output:

```bash
# 1. Generate responses for each condition
python src/normal.py
python src/top1.py
python src/wrong.py      # Misrouted condition

# 2. Compute semantic entropy across all three conditions
python src/SemanticEntropy.py

# 3. Run statistical tests
python src/statTest.py

# 4. Generate visualisations
python src/visualisations.py
```

## Data Format

`data/prompts.jsonl` contains one prompt per line in the following format:

```json
{"category": "factual", "prompt": "Your prompt here"}
```

Three prompt categories are used: `factual`, `open_ended`, and `unanswerable`.

## Results

All outputs are saved to `results/`:

| File | Description |
|------|-------------|
| `normal.jsonl` / `misrouted.jsonl` / `top1.jsonl` | Raw model responses per condition |
| `per_prompt_comparison.csv/json` | Entropy scores per prompt across conditions |
| `per_category_comparison.csv/json` | Aggregated entropy per category |
| `statistical_results.json` | Full statistical test results |
| `bar_chart_entropy.png` | Mean entropy by category and condition |
| `bar_chart_clusters.png` | Mean cluster count by category and condition |
| `box_plots_entropy.png` | Entropy distributions per condition |
| `histogram_entropy.png` | Entropy histograms across all prompts |
| `scatter_entropy.png` | Per-prompt entropy: misrouted and top-1 vs baseline |
| `table_entropy.png` | Mean ± SD entropy summary table |
| `table_clusters.png` | Mean ± SD cluster count summary table |
| `table_descriptive.png` | Descriptive statistics table |
| `table_shapiro_wilk.png` | Normality test results and selected test per comparison |
| `table_test_results.png` | Final statistical test results |

## Attribution

- Qwen1.5-MoE-A2.7B-Chat: [Qwen Team, Hugging Face](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)
- 4-bit quantisation: Dettmers et al. (2023), QLoRA
- Sentence embeddings: `all-mpnet-base-v2` via [sentence-transformers](https://www.sbert.net)
- Semantic entropy methodology: Farquhar et al. (2024)
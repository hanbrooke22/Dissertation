import json
import torch
from model import model, tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
PROMPTS_PATH      = "data/prompts.jsonl"
OUTPUT_PATH       = "results/misrouted.jsonl"
MISROUTE_STRENGTH = 0.3   # 0.0 = no misrouting, 1.0 = fully random
                           # Start at 0.3 — increase if entropy diff too small,
                           # decrease if responses are still garbage
MAX_RETRIES       = 3     # Retries per prompt if too many empty responses
MIN_VALID         = 5     # Minimum non-empty responses required to accept a batch

torch.manual_seed(42)

# ── Register forward hooks on gate Linear layers ──────────────────────────────
# Qwen2MoeSparseMoeBlock uses self.gate = nn.Linear(hidden_size, num_experts)
# We blend real logits with noise: (1 - strength) * real + strength * noise
# Noise is scaled to the same mean/std as the real logits so magnitude doesn't
# dominate — only direction (i.e. expert selection) is disrupted.

hooks      = []
gate_count = 0

for name, module in model.named_modules():
    if "mlp.gate" in name and isinstance(module, torch.nn.Linear):
        if module.out_features > 1:   # skip shared_expert_gate (out_features=1)
            def make_hook(strength):
                def hook_fn(module, input, output):
                    noise = torch.rand_like(output)
                    noise = noise * output.std() + output.mean()
                    return (1.0 - strength) * output + strength * noise
                return hook_fn

            h = module.register_forward_hook(make_hook(MISROUTE_STRENGTH))
            hooks.append(h)
            gate_count += 1

print(f"Registered hooks on {gate_count} gate modules (strength={MISROUTE_STRENGTH})")

if gate_count == 0:
    print("WARNING: No gate modules found. Printing all Linear layers to diagnose:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}  (out_features={module.out_features})")
    raise RuntimeError("Fix the gate filter above and rerun.")

# ── Generation helper ─────────────────────────────────────────────────────────
def generate_responses(prompt, n=10):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always give a direct, confident answer in a full sentence."},
        {"role": "user",   "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_len = model_inputs.input_ids.shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            num_return_sequences=n,
            temperature=0.8,
            top_p=0.95
        )

    generated_ids = generated_ids[:, input_len:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# ── Main generation loop ──────────────────────────────────────────────────────
with open(PROMPTS_PATH, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH,  "w", encoding="utf-8") as f_out:

    for line in f_in:
        line = line.strip()
        if not line:
            continue

        item   = json.loads(line)
        prompt = item["prompt"]

        responses = []
        for attempt in range(1, MAX_RETRIES + 1):
            candidates = generate_responses(prompt)
            valid      = [r for r in candidates if len(r.strip()) > 5]
            if len(valid) >= MIN_VALID:
                responses = candidates
                break
            print(f"  Attempt {attempt}/{MAX_RETRIES}: "
                  f"only {len(valid)} valid — retrying...")

        if len([r for r in responses if len(r.strip()) > 5]) < MIN_VALID:
            print(f"  WARN: gave up on '{prompt[:50]}...' — "
                  f"lower MISROUTE_STRENGTH if this keeps happening")

        out_item = {
            "category":  item.get("category"),
            "prompt":    prompt,
            "responses": responses
        }
        f_out.write(json.dumps(out_item) + "\n")

        n_valid = len([r for r in responses if len(r.strip()) > 5])
        print(f"[{n_valid}/10 valid] {prompt[:70]}")

# ── Clean up ──────────────────────────────────────────────────────────────────
for h in hooks:
    h.remove()

print("\nDone. Hooks removed.")
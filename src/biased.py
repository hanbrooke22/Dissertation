import json, torch, random
import numpy as np
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from model import model, tokenizer

prompts_path = "data/prompts.jsonl"
output_path = "results/biased.jsonl"

# Qwen1.5-MoE uses Qwen2MoeTopKRouter which runs:
#   1. softmax on gate logits
#   2. topk(router_logits, self.top_k)  <- top_k=4 normally
#
# We monkey-patch the forward method of each router to force top_k=2,
# keeping everything else identical to the original forward pass.
# This is the only clean way to change k without corrupting activations.

def make_top2_forward(original_forward, module):
    def patched_forward(hidden_states):
        hidden_states_flat = hidden_states.reshape(-1, module.hidden_dim)
        router_logits = F.linear(hidden_states_flat, module.weight)
        router_logits_softmax = torch.nn.functional.softmax(
            router_logits, dtype=torch.float, dim=-1
        )

        # Force top-2 instead of the default top-4
        router_top_value, router_indices = torch.topk(
            router_logits_softmax, k=2, dim=-1
        )

        if module.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )

        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value

        return router_logits, router_scores, router_indices
    return patched_forward


# Find and patch every router in the model
patched_modules = []
for name, module in model.named_modules():
    if hasattr(module, 'top_k') and hasattr(module, 'norm_topk_prob') and hasattr(module, 'hidden_dim'):
        original_forward = module.forward
        module.forward = make_top2_forward(original_forward, module)
        patched_modules.append((name, module, original_forward))

print(f"Patched {len(patched_modules)} router modules")


def generate_responses(prompt):
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
            num_return_sequences=10,
            temperature=0.8,
            top_p=0.95
        )

    generated_ids = generated_ids[:, input_len:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


with open(prompts_path, "r", encoding="utf-8") as f_in, \
     open(output_path,  "w", encoding="utf-8") as f_out:

    for line in f_in:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        prompt = item["prompt"]

        responses = generate_responses(prompt)

        out_item = {
            "category":  item.get("category"),
            "prompt":    prompt,
            "responses": responses
        }
        f_out.write(json.dumps(out_item) + "\n")

# Restore original forward methods
for name, module, original_forward in patched_modules:
    module.forward = original_forward

print("Restored all router modules")
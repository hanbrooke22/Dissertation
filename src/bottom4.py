import json, torch, random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from model import model, tokenizer

prompts_path = "data/prompts.jsonl"
output_path = "results/misrouted_bottom.jsonl"

hooks = []

# Loop through every layer of the model, look for layers that decide which expert to activate
for name, module in model.named_modules():
    if "mlp.gate" in name and isinstance(module, torch.nn.Linear):
        if module.out_features > 1:

            # Negate the gate logits so the bottom 4 experts become the top 4
            def make_hook():
                def hook_fn(module, input, output):
                    return -output
                return hook_fn

            # Attach hook to each gate layer, save so it can be removed later
            h = module.register_forward_hook(make_hook())
            hooks.append(h)


def generate_responses(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always give a direct, confident answer in a full sentence."},
        {"role": "user",   "content": prompt}
    ]

    # Format the prompt into the chat format and tokenize
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

    # Strip the input tokens and decode back to text
    generated_ids = generated_ids[:, input_len:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# Output to the file
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

# Remove all hooks so the model returns to normal
for h in hooks:
    h.remove()
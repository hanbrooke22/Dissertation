import json, torch, random
import numpy as np

# Pin all the random bits to the same number so we get the same results every run
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from model import model, tokenizer

prompts_path = "data/prompts.jsonl"
output_path = "results/misrouted.jsonl" # Where the answers go when we mess with the routing

misroute_strength = 0.4  # How much to corrupt the routing — 40% noise mixed into each gate output

# Hook into every MoE gate layer so we can interfere with its output as the model runs
hooks = []

for name, module in model.named_modules():
    if "mlp.gate" in name and isinstance(module, torch.nn.Linear):
        if module.out_features > 1:

            def make_hook(strength):
                # Inside the hook: blend the gate's normal output with random noise, with the strength setting controlling the mix
                def hook_fn(module, input, output):
                    noise = torch.rand_like(output)
                    noise = noise * output.std() + output.mean()
                    return (1.0 - strength) * output + strength * noise
                return hook_fn

            # Attach the hook to this layer and keep hold of it so we can take it back off afterwards
            h = module.register_forward_hook(make_hook(misroute_strength))
            hooks.append(h)

def generate_responses(prompt):
    # Get 10 different answers for one prompt with the misrouted gates in place
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always give a direct, confident answer in a full sentence."},
        {"role": "user",   "content": prompt}
    ]

    # Wrap the messages up in the format the model expects to see
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

    # Chop the original prompt off the front so we're left with just the model's reply
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
        # Write each prompt and its 10 answers as one line in the output file
        f_out.write(json.dumps(out_item) + "\n")

# Take all the hooks off so the model's back to normal for anything else that uses it
for h in hooks:
    h.remove()
import json, torch, random
import numpy as np

# Pin all the random bits to the same number so we get the same results every run
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from model import model, tokenizer

prompts_path = "data/prompts.jsonl"
output_path = "results/normal.jsonl" # Where the answers get saved

with open(prompts_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        prompt = item["prompt"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always give a direct, confident answer in a full sentence"},
            {"role": "user", "content": prompt}
        ]

        # Wrap the messages up in the format the model expects to see
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)
        input_len = model_inputs.input_ids.shape[1]

        # Ask the model for 10 different responses to the same prompt so we can see how much it varies
        with torch.inference_mode(): # Tells PyTorch we're not training, just running, so it can skip the extra bookkeeping
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
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        out_item = {
            "category": item.get("category"),
            "prompt": item.get("prompt"),
            "responses": responses
        }

        # Write each prompt and its 10 answers as one line in the output file
        f_out.write(json.dumps(out_item) + "\n")
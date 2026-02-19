import json, torch, time
from model import model, tokenizer

prompts_path = "data/prompts.jsonl"
normal_out_path = "results/normal.jsonl"

with open(prompts_path, "r", encoding="utf-8") as f_in, open(normal_out_path, "w", encoding="utf-8") as f_out:
    for line in f_in: 
        line = line.strip()
        if not line: 
            continue

        item = json.loads(line)
        prompt = item["prompt"]

        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = model_inputs.input_ids.shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=True, 
                num_return_sequences=5,
                temperature=0.8, 
                top_p=0.95
            )

        generated_ids = generated_ids[:, input_len:]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        out_item = {
            "category": item.get("category"), 
            "prompt": item.get("prompt"), 
            "responses": responses
        }

        f_out.write(json.dumps(out_item) + "\n")

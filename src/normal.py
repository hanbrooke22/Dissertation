import json, os
from datetime import datetime
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

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        out_record = {
            "category": item.get("category"), 
            "prompt": item.get("prompt"), 
            "response": response
        }

        f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
import json
import torch
from model import model, tokenizer

torch.manual_seed(42)

prompt = "What is the capital of France?"

messages = [
    {"role": "system", "content": "You are a helpful assistant. Always give a direct, confident answer in a full sentence."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)
input_len = inputs.input_ids.shape[1]

with torch.inference_mode():
    ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        num_return_sequences=10,
        temperature=0.8,
        top_p=0.95
    )

ids = ids[:, input_len:]
responses = tokenizer.batch_decode(ids, skip_special_tokens=True)

print(json.dumps({"category": "single_answer", "prompt": prompt, "responses": responses}))
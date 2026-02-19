import json, torch
from model import model, tokenizer

prompts_path = "data/prompts.jsonl"
wrong_out_path = "results/misrouted.jsonl"

original_routes = {}

#Source: github
for name, module in model.named_modules():
    if hasattr(module, 'route_tokens_to_experts'):
        original_routes[name] = module.route_tokens_to_experts

        def make_misrouted(original_fn):
            def misrouted_routing(hidden_states, router_logits):
                selected_experts, routing_weights = original_fn(hidden_states, router_logits)
                #source:hugging face
                random_experts = torch.randint(
                    0, 
                    model.config.num_experts,
                    selected_experts.shape,
                    device=selected_experts.device
                )
                return random_experts, routing_weights
            return misrouted_routing
        
        module.route_tokens_to_experts = make_misrouted(original_routes[name])

with open(prompts_path, "r", encoding="utf-8") as f_in, open(wrong_out_path, "w", encoding="utf-8") as f_out:
    for line in f_in: 
        line = line.strip()
        if not line: 
            continue

        item = json.loads(line)
        prompt = item["prompt"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always answer in full sentences."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        input_len = model_inputs.input_ids.shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True, 
                num_return_sequences=10,
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mlx-community/Qwen1.5-MoE-A2.7B-4bit"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
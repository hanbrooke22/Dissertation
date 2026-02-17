import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen1.5-MoE-A2.7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
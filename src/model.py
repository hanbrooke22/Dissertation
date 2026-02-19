import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
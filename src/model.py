import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

# Settings for shrinking the model down so it fits in less memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Pull the model down and let it sort out which bits go on the GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# Grab the matching tokeniser so we can turn text into numbers the model understands
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
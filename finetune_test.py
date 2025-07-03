import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling
from datasets import load_dataset
import tqdm
import os 
import wandb

print("Imports done")
model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("loading dataset")
ds = load_dataset("winglian/tiny-shakespeare", cache_dir=cache_str)
print("loaded dataset")

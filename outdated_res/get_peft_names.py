from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling
import tqdm
import os 
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import torch

#paths
model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/phi4"

#calling model + cuda
base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                  torch_dtype=torch.float16, 
                                                  cache_dir=cache_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

#wandb
wandb.init(entity= "hdiaz-harvard-university", project="training-opwmth")
wandb.watch(base_model) 

#loading data
dataset_dict = load_dataset("open-web-math/open-web-math")
full_dataset = dataset_dict["train"]
subset = full_dataset.shuffle(seed=42).select(range(1000))

split = subset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

# Check results of loaded data
print(f"Subset size: {len(subset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str,)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def tokenize_function(examples):
#calll that pre-trained tokenizer
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train_data = train_dataset.map(tokenize_function, batched=True)
tokenized_test_data = test_dataset.map(tokenize_function, batched=True)


for name, module in base_model.named_modules():
    print(name)
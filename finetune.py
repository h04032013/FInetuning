#training on 1 single gpu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling
from datasets import load_dataset
import wandb
import tqdm
from peft import LoraConfig, get_peft_model, TaskType

print("Settings paths")
#Qwen/Qwen2.5-0.5B-Instruct
#microsoft/Phi-4-mini-instruct
model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub"
max_length=1024

print("loading models")
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_str)
print("defining device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("calling to.device")
base_model.to(device)
print("at to.device")

print("about to load dataset")
dataset = load_dataset("open-web-math/open-web-math",cache_dir=cache_str )
print("loaded dataset, about to make subset")
subset = dataset["train"].shuffle(seed=42).select(range(1000))

split = subset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

# Check results of loaded data 
print(f"Subset size: {len(subset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def tokenize(examples):
    return tokenizer(examples["text"], max_length=max_length, truncation=True)

tokenized_train_data = train_dataset.map(tokenize, batched=True)
tokenized_test_data = test_dataset.map(tokenize, batched=True)

#tokenized_train_data.set_format("torch")
#tokenized_test_data.set_format("torch")

tokenized_train_data.set_format(type="torch", columns=["input_ids","attention_mask"])
tokenized_test_data.set_format(type="torch", columns=["input_ids","attention_mask"])

print("TRAIN[0] SAMPLE", tokenized_train_data[0])
print("TEST[0] SAMPLE", tokenized_test_data[0])

wandb.init(entity= "hdiaz-harvard-university", project="training-opwmth")
#wandb.watch(base_model) 

print("setting lora_config")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],  # adjust for Phi architecture
    lora_dropout=0.001,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)

#qwen worked with batch_size=16, phi worked with batch_size=4, crashes at 8
training_args = TrainingArguments(
    output_dir=ft_cache,
    eval_strategy="epoch",
    learning_rate = 2e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #just 1 epoch for quick debugging
    num_train_epochs=1,
    weight_decay=1e-6,
    save_strategy="steps",
    save_steps=500,
   # logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
    report_to="wandb",
    run_name="ft-opwmth"
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(ft_cache)
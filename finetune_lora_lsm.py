#training for 1 gpu, using loss-masking logic due to phi input format
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer as HFTrainer
import os 
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import torch


model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/phi4"
os.makedirs("./grads", exist_ok=True)


base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, cache_dir=cache_str
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

#  wandb
wandb.init(entity="hdiaz-harvard-university", project="training-opwmth")
wandb.watch(base_model)

# Load dataset
dataset_dict = load_dataset("open-web-math/open-web-math")
full_dataset = dataset_dict["train"]
subset = full_dataset.shuffle(seed=42).select(range(1000))
split = subset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

print(f"Subset size: {len(subset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Loss-masked tokenizer
def tokenize_with_loss_mask(example):
    full_text = example["text"]
    assistant_tag = "<|assistant|>"

    # Tokenize the full input
    tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"][0]
    labels = input_ids.clone()

    # Find index right after <|assistant|>
    assistant_token_ids = tokenizer(assistant_tag, add_special_tokens=False)["input_ids"]
    match_idx = 0
    for i in range(len(input_ids) - len(assistant_token_ids)):
        if input_ids[i:i+len(assistant_token_ids)].tolist() == assistant_token_ids:
            match_idx = i + len(assistant_token_ids)
            break

    labels[:match_idx] = -100  # Mask everything before assistant response

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": tokenized["attention_mask"][0]
    }

# Tokenize datasets with loss masking
tokenized_train_data = train_dataset.map(tokenize_with_loss_mask)
tokenized_test_data = test_dataset.map(tokenize_with_loss_mask)
tokenized_train_data.set_format(type="torch")
tokenized_test_data.set_format(type="torch")

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.001,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

base_model.config.use_cache = False

model = get_peft_model(base_model, lora_config)
model.train()
model.gradient_checkpointing_enable()

# Custom Trainer for gradient logging
class GradientSavingTrainer(HFTrainer):
    def training_step(self, model, inputs, batch_size): 
        loss = super().training_step(model, inputs, batch_size)
        if self.state.global_step % 500 == 0:
            save_path = f"./grads/step_{self.state.global_step}"
            os.makedirs(save_path, exist_ok=True)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    torch.save(param.grad.clone().cpu(), f"{save_path}/{name.replace('.', '_')}_grad.pt")
                    if wandb.run is not None:
                        wandb.log(
                            {f"gradients/{name}": wandb.Histogram(param.grad.cpu().data.numpy())},
                            step=self.state.global_step
                        )
        return loss


#print("Transformers version:", transformers.__version__)
#print("TrainingArguments source:", TrainingArguments.__module__)

# Training arguments
training_args = TrainingArguments(
    output_dir=cache_str,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=1e-6,
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
    report_to="wandb",
    run_name="ft-opwmth"
)

# Initialize trainer
trainer = GradientSavingTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train and save
trainer.train()
model.save_pretrained(ft_cache)
tokenizer.save_pretrained(ft_cache)

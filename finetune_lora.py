from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling
import tqdm
import os 
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import torch


model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/phi4"
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)


wandb.init(entity= "hdiaz-harvard-university", project="training-opwmth")
wandb.watch(base_model) 
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

class GradientSavingTrainer(Trainer):
    def training_step(self, model, inputs, batch_size):

#Standard training step
        loss = super().training_step(model, inputs, batch_size)

        # Save gradients if needed
        if self.state.global_step % 500 == 0:  # every 500 steps
            save_path = f"./grads/step_{self.state.global_step}"
            os.makedirs(save_path, exist_ok=True)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    torch.save(param.grad.clone().cpu(), f"{save_path}/{name.replace('.', '_')}_grad.pt")
                    if wandb.run is not None:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().data.numpy())},
                                  step=self.state.global_step)


        return loss

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # adjust for Phi architecture
    lora_dropout=0.001,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
#model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=cache_str,
    eval_strategy="epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    #just 1 epoch for quick debugging
    num_train_epochs=1,
    weight_decay=0.000001,
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
    report_to="wandb",
    run_name="ft-opwmth"
    )

trainer = GradientSavingTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(output_dir=ft_cache)
import torch
import os
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets
import tqdm
import wandb
from itertools import islice
import shutil

print("Imports done")
model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub"

# Create cache directories if they don't exist
os.makedirs(cache_str, exist_ok=True)
os.makedirs(ft_cache, exist_ok=True)

# Create a temporary directory for dataset caching
temp_cache_dir = tempfile.mkdtemp()
os.environ['HF_DATASETS_CACHE'] = temp_cache_dir

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#print("Attributes done, to device done")

#print("Initializing wandb")
wandb.init(entity="hdiaz-harvard-university", project="training-opwmth")
wandb.watch(model)  

#print("now loading dataset")
print("Loading dataset...")
try:
    # Load dataset with temporary cache
    dataset = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        cache_dir=temp_cache_dir,
        download_mode="force_redownload"
    )

    # Take a smaller subset for testing
    subset_size = int(0.05 * len(dataset))  # 5% of the data
    dataset = dataset.shuffle(seed=42).select(range(subset_size))

    # Split into train and test
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Check results
    print(f"Full dataset size: {len(dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train_data = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )

    tokenized_test_data = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=test_dataset.column_names
    )

    tokenized_train_data.set_format("torch")
    tokenized_test_data.set_format("torch")

    print(next(iter(tokenized_test_data)))
    print(next(iter(tokenized_test_data)))

    #print(tokenized_test_data.columns)
    #print(tokenized_train_data.columns)

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

    training_args = TrainingArguments(
        output_dir=cache_str,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        push_to_hub=False,
        report_to="wandb",
        run_name="ft-opwmth",
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4  # Accumulate gradients to reduce memory usage
    )

    trainer = GradientSavingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    #training
    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir=ft_cache)

finally:
    # Clean up temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_cache_dir, ignore_errors=True)


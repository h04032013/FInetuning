import torch
import os
import wandb
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.distributed import is_initialized, get_rank

def main():
    print("Starting distributed training")

    # Setup environment (optional but helps)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    
    model_name = "microsoft/Phi-4-mini-instruct"
    cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
    ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_str
    ).to(device)
    
    # Only log to wandb on process rank 0
    if local_rank == 0:
        wandb.init(entity="hdiaz-harvard-university", project="training-opwmth")
        wandb.watch(model)
    else:
        wandb.init(mode="disabled")
    
    dataset_dict = load_dataset("open-web-math/open-web-math")
    full_dataset = dataset_dict["train"]
    subset = full_dataset.shuffle(seed=42).select(range(1000))
    split = subset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_train_data = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_data = test_dataset.map(tokenize_function, batched=True)
    tokenized_train_data.set_format("torch")
    tokenized_test_data.set_format("torch")

    class GradientSavingTrainer(Trainer):
        def training_step(self, model, inputs, batch_size):
            loss = super().training_step(model, inputs, batch_size)
            if self.state.global_step % 500 == 0 and local_rank == 0:
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
        output_dir="./outputs",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,           # 4 per GPU
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,           # accumulate over 2 steps
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        weight_decay=0.000001,
        warmup_ratio=0.1,
        save_strategy="steps",
        save_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        report_to="wandb",
        run_name="ft-opwmth",
        fp16_full_eval=True,
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False
    )

    torch.cuda.set_device(local_rank)

    trainer = GradientSavingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    if local_rank == 0:
        trainer.save_model(output_dir=ft_cache)

if __name__ == "__main__":
    main()

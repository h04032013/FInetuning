import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, Any
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for fine-tuning parameters"""
    
    # Model configuration
    model_name = "microsoft/Phi-4-mini"
    dataset_name = "open-web-math/open-web-math"
    
    # LoRA configuration
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Training configuration
    output_dir = "./phi4-openwebmath-lora"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    max_grad_norm = 1.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    
    # Data configuration
    max_seq_length = 2048
    dataset_subset_size = 100000  # Use subset for faster training, set to None for full dataset
    
    # Logging and evaluation
    logging_steps = 10
    eval_steps = 500
    save_steps = 1000
    evaluation_strategy = "steps"
    save_strategy = "steps"
    load_best_model_at_end = True
    metric_for_best_model = "eval_loss"
    greater_is_better = False
    
    # Weights & Biases logging (optional)
    use_wandb = True
    wandb_project = "phi4-openwebmath-lora"


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Setup tokenizer with proper configuration"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def setup_model(model_name: str, lora_config: LoraConfig) -> AutoModelForCausalLM:
    """Setup model with LoRA configuration"""
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Disable cache for training
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def prepare_dataset(dataset_name: str, tokenizer: AutoTokenizer, config: Config):
    """Load and preprocess the OpenWebMath dataset"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Take subset if specified
    if config.dataset_subset_size:
        dataset = dataset.take(config.dataset_subset_size)
    
    # Convert to regular dataset for easier processing
    dataset = dataset.to_iterable_dataset()
    
    def tokenize_function(examples):
        """Tokenize the text data"""
        # The OpenWebMath dataset has a 'text' field
        texts = examples['text']
        
        # Tokenize with truncation and padding
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.max_seq_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset for training and evaluation
    train_dataset = tokenized_dataset.take(int(0.95 * (config.dataset_subset_size or 100000)))
    eval_dataset = tokenized_dataset.skip(int(0.95 * (config.dataset_subset_size or 100000)))
    
    return train_dataset, eval_dataset


def main():
    """Main training function"""
    config = Config()
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=vars(config))
    
    # Setup tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = setup_tokenizer(config.model_name)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Setup model
    logger.info("Setting up model with LoRA...")
    model = setup_model(config.model_name, lora_config)
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(config.dataset_name, tokenizer, config)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked language modeling
        pad_to_multiple_of=8
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=True,  # Use mixed precision training
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="wandb" if config.use_wandb else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save LoRA adapters specifically
    model.save_pretrained(config.output_dir)
    
    logger.info(f"Training completed! Model saved to {config.output_dir}")
    
    # Finish wandb run
    if config.use_wandb:
        wandb.finish()


def load_trained_model(model_path: str, base_model_name: str):
    """
    Function to load the trained LoRA model for inference
    """
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import transformers
        import datasets
        import peft
        import torch
        print("All required packages are available!")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install transformers datasets peft torch accelerate wandb")
        exit(1)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available. Training will be very slow on CPU.")
    
    # Run training
    main()
# train.py
from torchtune.models.phi3 import lora_phi3_mini
from torchtune.data import build_tokenized_dataset
from torchtune.trainers import train_lora
from torchtune.utils.config import TuneConfig
import torch
import wandb
from transformers import AutoTokenizer

# Load config
config = TuneConfig.load("/n/netscratch/dam_lab/Lab/hdiaz/ft_project/torchtune/configs/phi4/lora.yaml")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = build_tokenized_dataset(
    path=config.dataset.path,
    split=config.dataset.split,
    tokenizer=tokenizer,
    max_length=config.dataset.max_length,
    num_samples=config.dataset.num_samples,
    train_val_split=config.dataset.train_val_split,
    shuffle=config.dataset.shuffle
)

train_dataset = dataset["train"]
eval_dataset = dataset["eval"]

# Create LoRA-wrapped model
model = lora_phi3_mini(
    pretrained_model_name=config.tokenizer.name,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.001,
    apply_lora_to_mlp=False,
    apply_lora_to_output=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optional: initialize W&B
if config.trainer.log_to_wandb:
    wandb.init(
        project=config.trainer.wandb_project,
        entity=config.trainer.wandb_entity,
        name=config.trainer.run_name
    )

# Start training
train_lora(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    config=config.trainer,
)

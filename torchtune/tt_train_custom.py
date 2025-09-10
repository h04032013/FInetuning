# train.py
import os, random, torch, wandb
from transformers import AutoTokenizer
from torchtune.utils.config import TuneConfig
from torchtune.data import build_tokenized_dataset
from torchtune.trainers import train_lora

# ---- NEW: custom Phi-4-mini builder via phi3 components ----
from torchtune.models.phi3._component_builders import lora_phi3
from torchtune.training import FullModelHFCheckpointer

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def lora_phi4_mini(
    lora_attn_modules,
    apply_lora_to_mlp=False,
    apply_lora_to_output=False,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.0,
    use_dora=False,
    quantize_base=False,
):
    # Architecture taken from HF config.json for microsoft/Phi-4-mini-instruct
    # vocab_size=200064, n_layers=32, n_heads=24, n_kv_heads=8, hidden=3072, intermed=8192, max_seq_len=131072
    return lora_phi3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=200_064,
        num_layers=32,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        intermediate_dim=8192,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

def main():
    # Load YAML
    config = TuneConfig.load("/n/netscratch/dam_lab/Lab/hdiaz/ft_project/torchtune/configs/phi-4-mini/lora_custom.yaml")

    set_seed(getattr(config.trainer, "seed", 42))

    # --- Tokenizer (HF) ---
    # MS sample recommends pad=unk to avoid endless generation
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name, trust_remote_code=True)
    tokenizer.model_max_length = config.dataset.max_length
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"

    # --- Dataset ---
    dataset = build_tokenized_dataset(
        path=config.dataset.path,
        split=config.dataset.split,
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
        num_samples=config.dataset.num_samples,
        train_val_split=config.dataset.train_val_split,
        shuffle=config.dataset.shuffle,
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    # --- Model (custom builder) ---
    model = lora_phi4_mini(
        lora_attn_modules=config.lora.target_modules,   # e.g. ["q_proj","k_proj","v_proj","output_proj"]
        apply_lora_to_mlp=getattr(config.lora, "apply_lora_to_mlp", False),
        apply_lora_to_output=getattr(config.lora, "apply_lora_to_output", False),
        lora_rank=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
    )

    # --- Load HF checkpoint weights into torchtune model ---
    # First download weights locally once (e.g. with: `tune download microsoft/Phi-4-mini-instruct --output-dir /path`)
    ckpt = FullModelHFCheckpointer(
        checkpoint_dir=config.checkpointer.checkpoint_dir,            # local dir containing the safetensors
        checkpoint_files=config.checkpointer.checkpoint_files,        # ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        model_type="PHI4",                                            # torchtune model type string
        output_dir=config.trainer.output_dir,
    )
    state = ckpt.load_checkpoint()            # returns a torchtune-formatted state_dict
    model.load_state_dict(state["model"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Optional: W&B ---
    if config.trainer.log_to_wandb:
        wandb.init(
            project=config.trainer.wandb_project,
            entity=config.trainer.wandb_entity,
            name=config.trainer.run_name,
            config={
                "lr": config.trainer.learning_rate,
                "epochs": config.trainer.num_train_epochs,
                "batch_train": config.trainer.per_device_train_batch_size,
                "batch_eval": config.trainer.per_device_eval_batch_size,
                "max_length": config.dataset.max_length,
                "lora_rank": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "targets": config.lora.target_modules,
            },
        )

    # --- Train ---
    train_lora(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=config.trainer,    # your trainer block
    )

if __name__ == "__main__":
    main()

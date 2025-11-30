import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk


def setup_model_and_tokenizer(model_name="gpt2"):
    """Load base model + tokenizer (Mac-friendly, no bitsandbytes)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Determine device - use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS has issues with fp16 in training
        print("MPS is enabled")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("MPS is disabled")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,  # Don't use auto device map
        torch_dtype=dtype,
    ).to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # LoRA setup (reduced rank for faster training)
    lora_cfg = LoraConfig(
        r=8,  # Reduced from 16 for faster training
        lora_alpha=16,  # Typically 2x the rank
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def train_model():
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()

    print("Loading dataset...")
    dataset = load_from_disk("gender_bias_dataset")

    # tokenize the formatted_text column up front
    def tokenize_batch(batch):
        tok = tokenizer(
            batch["formatted_text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Reduced to 128 for memory constraints
        )
        # Copy input_ids to labels for causal language modeling
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir="./gender-bias-gpt2",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,  # Disable fp16 for MPS compatibility
        save_total_limit=2,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("🚀 Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("./gender-bias-gpt2-final")
    tokenizer.save_pretrained("./gender-bias-gpt2-final")
    print("Training complete! Model saved to ./gender-bias-gpt2-final")


if __name__ == "__main__":
    train_model()

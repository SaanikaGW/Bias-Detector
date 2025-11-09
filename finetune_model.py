import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk


def setup_model_and_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load base model + tokenizer (Mac-friendly, no bitsandbytes)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
    )

    # LoRA setup
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],
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
        return tokenizer(
            batch["formatted_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["formatted_text"])

    training_args = TrainingArguments(
        output_dir="./gender-bias-llama",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=torch.backends.mps.is_available(),
        save_total_limit=2,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
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
    trainer.save_model("./gender-bias-llama-final")
    tokenizer.save_pretrained("./gender-bias-llama-final")
    print("Training complete! Model saved to ./gender-bias-llama-final")


if __name__ == "__main__":
    train_model()

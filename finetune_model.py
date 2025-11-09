import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "gender_bias_dataset"

def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 👇 FORCE CPU, no auto, no MPS
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,
        torch_dtype=torch.float32,
    ).to("cpu")

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
    print("Loading model and tokenizer on CPU...")
    model, tokenizer = setup_model_and_tokenizer()

    print("Loading dataset from disk...")
    ds = load_from_disk(DATA_PATH)

    def tokenize_batch(batch):
        tok = tokenizer(
            batch["formatted_text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        tok["labels"] = tok["input_ids"]
        return tok

    print("Tokenizing dataset and adding labels...")
    ds = ds.map(tokenize_batch, batched=True)
    print("Train columns:", ds["train"].column_names)

    training_args = TrainingArguments(
        output_dir="./gender-bias-llama",
        per_device_train_batch_size=1,     # CPU-safe
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        no_cuda=True,                      # 👈 tell Trainer to stay off GPU/MPS
        dataloader_pin_memory=False,       # no MPS pin warning
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
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )

    print("🚀 Starting training on CPU...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("./gender-bias-llama-final")
    tokenizer.save_pretrained("./gender-bias-llama-final")
    print("✅ Done! trained on CPU")

if __name__ == "__main__":
    train_model()


import os
from datasets import Dataset, DatasetDict
import pandas as pd

CSV_PATH = "gender_bias_dataset_final_fixed.csv"
COLS = ["Text", "Gender_bias_sentiment", "Bias_type", "Bias_explanation"]

def load_gender_bias_data(csv_path: str) -> pd.DataFrame:
    """
    Manually load the CSV and tolerate commas in the last column.
    We split each line into at most 4 parts.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # assume first line is header, skip it
    for idx, line in enumerate(lines[1:], start=2):
        # split into 4 pieces max
        parts = line.split(",", 3)
        if len(parts) < 4:
            # skip broken lines but tell the user
            print(f"Skipping bad line {idx}: {line}")
            continue

        text, sentiment, bias_type, bias_explanation = parts

        rows.append({
            "Text": text.strip(),
            "Gender_bias_sentiment": sentiment.strip().lower(),
            "Bias_type": bias_type.strip().lower(),
            # keep explanation as-is (it may contain commas)
            "Bias_explanation": bias_explanation.strip(),
        })

    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        raise ValueError("No valid rows were loaded from the CSV.")
    return df

def format_for_training(example):
    instruction = (
        "Analyze the following text for gender bias. Identify if bias exists, "
        "the type, sentiment, and explain where it appears in the text."
    )

    text_val = example["Text"]
    sentiment = example.get("Gender_bias_sentiment", "positive")
    bias_type = example.get("Bias_type", "none")
    explanation = example.get("Bias_explanation", "No explanation provided")

    if bias_type == "none":
        response = (
            "NO BIAS DETECTED: Statement presents gender in a neutral or factual "
            "way without implying bias or hierarchy."
        )
    else:
        response = (
            "BIAS DETECTED:\n"
            f"- Sentiment: {sentiment}\n"
            f"- Type: {bias_type}\n"
            f"- Explanation: {explanation}"
        )

    formatted = f"""<s>[INST] {instruction}

Text: {text_val} [/INST]

{response}</s>"""
    return {"formatted_text": formatted}

def prepare_dataset(csv_file_path: str):
    df = load_gender_bias_data(csv_file_path)

    print(f"Loaded {len(df)} examples")
    print("Bias sentiment distribution:", df["Gender_bias_sentiment"].value_counts().to_dict())
    print("Bias types:", df["Bias_type"].value_counts().to_dict())

    # to Hugging Face dataset
    dataset = Dataset.from_pandas(df)

    # add formatted text
    dataset = dataset.map(format_for_training)

    # split
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": train_test["test"],
        }
    )

    dataset_dict.save_to_disk("gender_bias_dataset")
    print(
        f"Dataset saved: {len(dataset_dict['train'])} training, "
        f"{len(dataset_dict['validation'])} validation examples"
    )

    return dataset_dict

if __name__ == "__main__":
    prepare_dataset(CSV_PATH)

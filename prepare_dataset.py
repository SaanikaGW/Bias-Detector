import os
from datasets import Dataset, DatasetDict
import pandas as pd

# 🔁 Set this to your actual file name
# e.g. "sample_data - Sheet1.csv" or "gender_bias_dataset_final_fixed.csv"
CSV_PATH = "bios_check_dataset_full.csv"

# Now includes Biased_span
COLS = ["Text", "Gender_bias_sentiment", "Bias_type", "Biased_span", "Bias_explanation"]

# All known bias categories (including new ones in the full dataset)
KNOWN_CATEGORIES = {
    "explicit_gender", "stereotype", "requirements_bias", "age_bias",
    "masculine_coded_language", "motherhood_penalty", "appearance_bias",
}


def load_gender_bias_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV exported from Google Sheets, including Biased_span.
    Drops stray 'Unnamed' columns and normalizes label text.
    """
    df = pd.read_csv(csv_path)

    # Drop any blank extra columns like 'Unnamed: 5'
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Make sure required columns exist
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Normalize text columns
    df["Text"] = df["Text"].astype(str).str.strip()

    df["Gender_bias_sentiment"] = (
        df["Gender_bias_sentiment"].astype(str).str.strip().str.lower()
    )
    df["Bias_type"] = df["Bias_type"].astype(str).str.strip().str.lower()

    # Biased_span can be NaN for neutral examples → replace with empty string
    df["Biased_span"] = df["Biased_span"].fillna("").astype(str).str.strip()

    # Explanation can be NaN → replace with generic
    df["Bias_explanation"] = (
        df["Bias_explanation"]
        .fillna("No explanation provided.")
        .astype(str)
        .str.strip()
    )

    # Reorder columns just to be consistent
    df = df[COLS]

    if df.empty:
        raise ValueError("No valid rows were loaded from the CSV.")

    return df


def format_for_training(example):
    """
    Turn each row into an instruction-style prompt/response pair for causal LM.
    Now includes the biased span explicitly.
    """
    instruction = (
        "Analyze the following text for gender bias. Identify if bias exists, "
        "the type, the sentiment, where it appears in the text (biased span), "
        "and briefly explain why it is biased."
    )

    text_val = example["Text"]
    sentiment = example.get("Gender_bias_sentiment", "neutral")
    bias_type = example.get("Bias_type", "none")
    biased_span = example.get("Biased_span", "")
    explanation = example.get("Bias_explanation", "No explanation provided.")

    if bias_type == "none":
        response = (
            "NO BIAS DETECTED:\n"
            "- Sentiment: neutral\n"
            "- Type: none\n"
            "- Biased span: N/A\n"
            "- Explanation: Statement presents gender in a neutral or factual "
            "way without implying bias or hierarchy."
        )
    else:
        # If span is empty but type != none, still give something
        span_str = biased_span if biased_span else "Span not specified."
        response = (
            "BIAS DETECTED:\n"
            f"- Sentiment: {sentiment}\n"
            f"- Type: {bias_type}\n"
            f"- Biased span: {span_str}\n"
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

    # convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)

    # add formatted text column used for training
    dataset = dataset.map(format_for_training)

    # split train/validation
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

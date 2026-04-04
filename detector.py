"""
detector.py — Core bias detection logic for BIOS Check Careers
Primary: fine-tuned GPT-2 LoRA model (gender-bias-gpt2-final)
Fallback: rule-based regex patterns + sentence-transformers
Bias spans are loaded from bios_check_dataset_full.csv at startup.
"""
import re
import string
import os
import pandas as pd
from typing import Optional, List

# ── Load biased spans from dataset ────────────────────────────────────────────

_CSV_PATH = os.path.join(os.path.dirname(__file__), "bios_check_dataset_full.csv")

def _load_spans_from_csv(csv_path: str) -> dict[str, list[str]]:
    spans: dict[str, list[str]] = {}
    try:
        df = pd.read_csv(csv_path)
        biased = df[df["Gender_bias_sentiment"] == "biased"]
        for _, row in biased.iterrows():
            cat  = str(row.get("Bias_type", "")).strip().lower()
            span = str(row.get("Biased_span", "")).strip()
            if cat and span and span.lower() not in ("nan", ""):
                spans.setdefault(cat, [])
                if span not in spans[cat]:
                    spans[cat].append(span)
        print(f"[detector] Loaded spans from CSV: { {k: len(v) for k, v in spans.items()} }")
    except Exception as e:
        print(f"[detector] Could not load CSV spans ({e}) — using built-in patterns only.")
    return spans

CSV_SPANS = _load_spans_from_csv(_CSV_PATH)

# ── Fine-tuned GPT-2 model (primary detector) ─────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gender-bias-gpt2-final")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _base = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(_device)
    _gpt2_model = PeftModel.from_pretrained(_base, MODEL_PATH).to(_device)
    _gpt2_model.eval()
    GPT2_AVAIL = True
    print("[detector] Fine-tuned GPT-2 model loaded.")
except Exception as e:
    GPT2_AVAIL = False
    print(f"[detector] Fine-tuned model unavailable ({e}) — using rule-based fallback.")

# ── Optional semantic layer ────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _EMB_MODEL     = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_AVAIL = True
except ImportError:
    SEMANTIC_AVAIL = False

# ── Category weights (all categories incl. new ones) ──────────────────────────

CATEGORY_WEIGHTS = {
    "explicit_gender":          1.0,
    "stereotype":               0.8,
    "masculine_coded_language": 0.8,
    "motherhood_penalty":       0.9,
    "appearance_bias":          0.7,
    "age_bias":                 0.7,
    "requirements_bias":        0.6,
}

# ── Rule-based fallback patterns ───────────────────────────────────────────────

BIAS_PATTERNS: dict[str, list[str]] = {
    "explicit_gender": [
        r"\bhe\b", r"\bshe\b", r"\bhim\b", r"\bher\b", r"\bhis\b",
        r"\bmale\b", r"\bfemale\b", r"\bgentlemen\b",
        r"female\s+candidate\s+preferred", r"male\s+candidate\s+preferred",
        r"prefer\s+female", r"prefer\s+male",
        r"young\s+man", r"young\s+woman",
        r"businessman", r"businesswoman", r"salesman", r"saleswoman",
        r"manpower", r"mankind", r"\bgirl\b", r"\bboy\b",
        r"chairman", r"spokeswoman", r"spokesman",
    ],
    "stereotype": [
        r"nurturing", r"aggressive\s+(leader|self.starter)", r"dominant\s+leader",
        r"assertive\s+man", r"bossy", r"emotional",
        r"manly", r"womanly", r"motherly", r"fatherly",
    ],
    "masculine_coded_language": [
        r"masculine", r"feminine", r"soft\s+skills",
        r"rock\s+star", r"ninja", r"dominant", r"aggressive",
        r"competitive\s+spirit", r"alpha",
    ],
    "motherhood_penalty": [
        r"no\s+family\s+commitments", r"no\s+caregiving\s+responsibilities",
        r"must\s+be\s+fully\s+dedicated", r"lifestyle\s+fit",
    ],
    "appearance_bias": [
        r"well[\s-]groomed", r"presentable", r"attractive",
        r"good[\s-]looking", r"physically\s+fit", r"clean[\s-]cut",
    ],
    "requirements_bias": [
        r"available\s+24[\/\s]7", r"no\s+exceptions",
        r"must\s+be\s+available\s+at\s+all\s+times",
        r"willing\s+to\s+relocate\s+immediately",
        r"must\s+travel\s+frequently",
        r"no\s+gaps\s+in\s+(employment|work)\s+history",
    ],
    "age_bias": [
        r"fresh\s+graduate\s+only", r"recent\s+graduate\s+preferred",
        r"young\s+(and\s+)?energetic", r"under\s+\d{2}(\s+preferred)?",
        r"\d{2}\s+years\s+or\s+younger", r"digital\s+native",
        r"new\s+grad", r"entry[\-\s]level.*no\s+experience",
    ],
}

SEMANTIC_REFS: dict[str, str] = {
    "explicit_gender":          "We prefer hiring a female or male candidate for this position.",
    "stereotype":               "Looking for a nurturing or aggressive dominant leader to fill this role.",
    "masculine_coded_language": "We want a rockstar ninja who is aggressive and dominant.",
    "motherhood_penalty":       "The role requires full dedication with no family commitments.",
    "appearance_bias":          "Candidate must be well-groomed, attractive and physically presentable.",
    "requirements_bias":        "Must be available 24/7 with no family commitments and willing to relocate immediately.",
    "age_bias":                 "We are looking for fresh graduates, young energetic candidates under 30.",
}

SEMANTIC_THRESHOLD = 0.72

INSTRUCTION = (
    "Analyze the following text for gender bias. Identify if bias exists, "
    "the type, the sentiment, where it appears in the text (biased span), "
    "and briefly explain why it is biased."
)


# ── CSV span matching ──────────────────────────────────────────────────────────

def csv_span_match(text: str, norm: str) -> list[dict]:
    """Match biased spans loaded directly from the dataset CSV."""
    highlights = []
    for category, spans in CSV_SPANS.items():
        for span in spans:
            span_norm = span.lower()
            if span_norm in norm:
                highlights.append({"span": span, "type": category, "source": "csv"})
    return highlights


# ── GPT-2 inference ────────────────────────────────────────────────────────────

def _run_gpt2(sentence: str) -> str:
    prompt = f"<s>[INST] {INSTRUCTION}\n\nText: {sentence} [/INST]\n\n"
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(_device)
    with torch.no_grad():
        output = _gpt2_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_gpt2_output(sentence: str, raw: str) -> Optional[List[dict]]:

    raw_lower = raw.lower()
    if "no bias detected" in raw_lower:
        return None

    type_match = re.search(r"-\s*type:\s*([^\n]+)", raw, re.IGNORECASE)
    bias_type  = type_match.group(1).strip().lower().replace(" ", "_") if type_match else "explicit_gender"
    if bias_type not in CATEGORY_WEIGHTS:
        if "gender" in bias_type:
            bias_type = "explicit_gender"
        elif "stereo" in bias_type:
            bias_type = "stereotype"
        elif "masculine" in bias_type:
            bias_type = "masculine_coded_language"
        elif "mother" in bias_type or "penalty" in bias_type:
            bias_type = "motherhood_penalty"
        elif "appear" in bias_type:
            bias_type = "appearance_bias"
        elif "age" in bias_type:
            bias_type = "age_bias"
        elif "require" in bias_type:
            bias_type = "requirements_bias"
        else:
            bias_type = "explicit_gender"

    span_match = re.search(r"-\s*biased span:\s*([^\n]+)", raw, re.IGNORECASE)
    span = span_match.group(1).strip() if span_match else sentence[:80]
    if span.lower() in ("n/a", "span not specified.", ""):
        span = sentence[:80]

    return [{"span": span, "type": bias_type, "source": "gpt2"}]


def gpt2_detect(text: str) -> list[dict]:
    if not GPT2_AVAIL:
        return []
    highlights = []
    sentences  = re.split(r"[.!?\n]", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 8:
            continue
        raw    = _run_gpt2(sent)
        result = _parse_gpt2_output(sent, raw)
        if result:
            highlights.extend(result)
    return highlights


# ── Rule-based helpers ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_exact_spans(text: str, norm: str) -> list[dict]:
    highlights = []
    for category, patterns in BIAS_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, norm):
                span = text[m.start():m.end()]
                if span:
                    highlights.append({"span": span, "type": category, "source": "pattern"})
    return highlights


def semantic_span_match(text: str) -> list[dict]:
    if not SEMANTIC_AVAIL:
        return []
    highlights = []
    sentences  = re.split(r"[.!?;\n]", text)
    ref_embeddings = {
        cat: _EMB_MODEL.encode(ref, convert_to_tensor=True)
        for cat, ref in SEMANTIC_REFS.items()
    }
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 8:
            continue
        sent_emb = _EMB_MODEL.encode(sent, convert_to_tensor=True)
        for cat, ref_emb in ref_embeddings.items():
            score = float(st_util.pytorch_cos_sim(sent_emb, ref_emb)[0][0])
            if score >= SEMANTIC_THRESHOLD:
                highlights.append({
                    "span": sent[:120], "type": cat,
                    "source": "semantic", "similarity": round(score, 3),
                })
    return highlights


# ── Main public function ───────────────────────────────────────────────────────

def analyze_jd(text: str) -> dict:
    norm       = normalize_text(text)
    seen_spans: set[str] = set()
    highlights: list[dict] = []

    def add(hits: list[dict]):
        for h in hits:
            if h["span"] not in seen_spans:
                highlights.append(h)
                seen_spans.add(h["span"])

    # 1. Fine-tuned GPT-2 (primary)
    add(gpt2_detect(text))

    # 2. CSV span matching (dataset-derived)
    add(csv_span_match(text, norm))

    # 3. Rule-based regex patterns
    add(find_exact_spans(text, norm))

    # 4. Semantic similarity (fallback enrichment)
    add(semantic_span_match(text))

    categories_found = list({h["type"] for h in highlights})
    raw_score = min(
        1.0,
        sum(CATEGORY_WEIGHTS.get(c, 0.5) for c in categories_found) / len(CATEGORY_WEIGHTS)
    ) if categories_found else 0.0

    bias_level = "low" if raw_score < 0.25 else "medium" if raw_score < 0.6 else "high"

    return {
        "bias_score": round(raw_score, 3),
        "bias_level": bias_level,
        "categories": categories_found,
        "highlights": highlights,
    }

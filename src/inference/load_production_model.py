import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# trỏ về ROOT của project
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "config2_run1"
)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    return tokenizer, model

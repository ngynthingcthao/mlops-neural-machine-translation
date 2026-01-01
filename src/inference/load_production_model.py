import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.model.transformer import TransformerTranslationModel

# =====================================================
# PROJECT ROOT
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# =====================================================
# OPUS (LOCAL ARTIFACT)
# =====================================================
OPUS_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "opus_config1"     # 🔥 ĐÚNG THEO THỰC TẾ CỦA BẠN
)

# =====================================================
# TRANSFORMER (LOCAL ARTIFACT)
# =====================================================
TRANSFORMER_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "transformer_config3"
)

VOCAB_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "vocab"
)

# =====================================================
# LOAD OPUS MODEL (LOCAL)
# =====================================================
def load_opus_model():
    if not os.path.isdir(OPUS_DIR):
        raise FileNotFoundError(f"Opus model dir not found: {OPUS_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        OPUS_DIR,
        local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        OPUS_DIR,
        local_files_only=True
    )

    model.eval()
    return tokenizer, model


# =====================================================
# LOAD TRANSFORMER MODEL
# =====================================================
def load_transformer_model(device="cpu"):
    # ----- LOAD VOCAB -----
    with open(os.path.join(VOCAB_DIR, "src_vocab.json"), encoding="utf-8") as f:
        src_vocab = json.load(f)
    with open(os.path.join(VOCAB_DIR, "tgt_vocab.json"), encoding="utf-8") as f:
        tgt_vocab = json.load(f)

    id2tgt = {int(v): k for k, v in tgt_vocab.items()}

    # ----- LOAD CHECKPOINT -----
    ckpt_path = os.path.join(TRANSFORMER_DIR, "transformer.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint["config"]

    model = TransformerTranslationModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, id2tgt

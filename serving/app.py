import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.inference.load_production_model import (
    load_opus_model,
    load_transformer_model
)
from src.training.config import MAX_LEN

# =====================================================
# INIT APP
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(title="NMT EN → VI Service")

# =====================================================
# TEMPLATE & STATIC
# =====================================================
templates = Jinja2Templates(directory="serving/templates")

# (Nếu sau này có CSS/JS riêng)
# app.mount("/static", StaticFiles(directory="serving/static"), name="static")

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODELS (ONCE)
# =====================================================
opus_tokenizer, opus_model = load_opus_model()
transformer_model, src_vocab, id2tgt = load_transformer_model(device)

opus_model.to(device)
transformer_model.to(device)

SOS_ID = src_vocab["<SOS>"]
EOS_ID = src_vocab["<EOS>"]
PAD_ID = src_vocab["<PAD>"]
UNK_ID = src_vocab["<UNK>"]

# =====================================================
# REQUEST SCHEMA
# =====================================================
class TranslateRequest(BaseModel):
    text: str
    model_type: str  # opus | transformer


# =====================================================
# ROOT → UI
# =====================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =====================================================
# API TRANSLATE
# =====================================================
def encode_src(sentence: str):
    tokens = sentence.lower().strip().split()
    ids = [src_vocab.get(t, UNK_ID) for t in tokens]
    ids = ids[:MAX_LEN]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(device)


def decode_ids(ids):
    words = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        words.append(id2tgt.get(i, "<UNK>"))
    return " ".join(words)


@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()

    if not text:
        return JSONResponse(
            {"translation": "Empty input", "model_used": "none"},
            status_code=400
        )

    # ============ OPUS ============
    if req.model_type == "opus":
        inputs = opus_tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(device)

        with torch.no_grad():
            output = opus_model.generate(
                **inputs,
                max_length=MAX_LEN
            )

        pred = opus_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return {
            "translation": pred,
            "model_used": "opus"
        }

    # ============ TRANSFORMER ============
    if req.model_type == "transformer":
        src = encode_src(text)
        ys = torch.tensor([[SOS_ID]], dtype=torch.long).to(device)

        with torch.no_grad():
            for _ in range(MAX_LEN):
                out = transformer_model(src, ys)
                next_token = out[:, -1].argmax(-1).unsqueeze(1)
                ys = torch.cat([ys, next_token], dim=1)

                if next_token.item() == EOS_ID:
                    break

        pred = decode_ids(ys.squeeze().tolist())

        return {
            "translation": pred,
            "model_used": "transformer"
        }

    return {
        "translation": "Invalid model_type",
        "model_used": "error"
    }

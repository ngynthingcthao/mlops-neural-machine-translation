from fastapi import FastAPI
from pydantic import BaseModel
import torch

# IMPORT TUYỆT ĐỐI TỪ SRC
from src.inference.load_production_model import load_model

app = FastAPI(title="NMT EN to VI Service")

# Load model production (local artifact)
tokenizer, model = load_model()
model.eval()

class Request(BaseModel):
    text: str

class Response(BaseModel):
    translation: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/translate", response_model=Response)
def translate(req: Request):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=64
        )

    pred = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return {"translation": pred}

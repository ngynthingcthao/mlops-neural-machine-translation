# src/preprocessing/preprocess_and_split.py
import json
import random
from pathlib import Path

MAX_LEN = 50
SEED = 42

def normalize(x: str) -> str:
    return x.lower().strip()

def load_raw(
    en_path="data/raw/en_sents.txt",
    vi_path="data/raw/vi_sents.txt"
):
    en = Path(en_path).read_text(encoding="utf-8").splitlines()
    vi = Path(vi_path).read_text(encoding="utf-8").splitlines()
    assert len(en) == len(vi), "EN & VI files must have same length"
    return en, vi

def build_pairs(en, vi):
    """
    Build EN -> VI translation pairs
    """
    pairs = []
    for e, v in zip(en, vi):
        e, v = normalize(e), normalize(v)

        if not e or not v:
            continue

        if len(e.split()) > MAX_LEN or len(v.split()) > MAX_LEN:
            continue

        pairs.append({
            "src": e,
            "tgt": v
        })

    return pairs

def split_and_save(pairs, outdir="data/processed"):
    random.seed(SEED)
    random.shuffle(pairs)   # shuffle 1 LẦN DUY NHẤT

    n = len(pairs)

    # ===== 90 / 5 / 5 =====
    train = pairs[: int(0.9 * n)]
    val   = pairs[int(0.9 * n) : int(0.95 * n)]
    test  = pairs[int(0.95 * n) :]

    Path(outdir).mkdir(parents=True, exist_ok=True)

    for name, split in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        with open(f"{outdir}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Dataset size: {n}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

if __name__ == "__main__":
    en, vi = load_raw()
    pairs = build_pairs(en, vi)
    split_and_save(pairs)

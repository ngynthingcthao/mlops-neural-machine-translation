# src/preprocessing/preprocess_transformer.py
import json
from pathlib import Path

MAX_LEN = 50
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

# ===================== VOCAB =====================
def build_vocab(sentences):
    vocab = {}
    idx = 0

    for tok in SPECIAL_TOKENS:
        vocab[tok] = idx
        idx += 1

    for s in sentences:
        for w in s.split():
            if w not in vocab:
                vocab[w] = idx
                idx += 1

    return vocab

def sentence_to_ids(sentence, vocab, add_sos_eos=False):
    tokens = sentence.split()

    if add_sos_eos:
        tokens = ["<SOS>"] + tokens + ["<EOS>"]

    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids = ids[:MAX_LEN]
    ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

    return ids

def load_pairs(split):
    with open(f"data/processed/{split}.json", encoding="utf-8") as f:
        return json.load(f)

# ===================== MAIN =====================
def main():
    # ===== BUILD VOCAB FROM TRAIN ONLY (NO LEAKAGE) =====
    train_pairs = load_pairs("train")

    src_sentences = [p["src"] for p in train_pairs]
    tgt_sentences = [p["tgt"] for p in train_pairs]

    src_vocab = build_vocab(src_sentences)
    tgt_vocab = build_vocab(tgt_sentences)

    # ===== SAVE VOCAB =====
    Path("artifacts/vocab").mkdir(parents=True, exist_ok=True)

    with open("artifacts/vocab/src_vocab.json", "w", encoding="utf-8") as f:
        json.dump(src_vocab, f, ensure_ascii=False, indent=2)

    with open("artifacts/vocab/tgt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tgt_vocab, f, ensure_ascii=False, indent=2)

    # ===== CONVERT ALL SPLITS USING TRAIN VOCAB =====
    for split in ["train", "val", "test"]:
        pairs = load_pairs(split)
        out = []

        for p in pairs:
            out.append({
                "input_ids": sentence_to_ids(p["src"], src_vocab),
                "labels": sentence_to_ids(p["tgt"], tgt_vocab, add_sos_eos=True)
            })

        with open(f"data/processed/{split}_transformer.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print("Transformer preprocessing done successfully.")

if __name__ == "__main__":
    main()

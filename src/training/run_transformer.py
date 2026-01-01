import os
import json
import torch
import torch.nn as nn
import mlflow
import wandb

from torch.utils.data import DataLoader
from datasets import Dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm

from model.transformer import TransformerTranslationModel
from training.config import MAX_LEN


# ===================== LOAD DATA =====================
def load_split(name):
    with open(f"data/processed/{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ===================== LOAD VOCAB =====================
def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab  # token -> id


def invert_vocab(vocab):
    return {int(v): k for k, v in vocab.items()}  # id -> token


def decode_ids(ids, id2word, eos_id=2, pad_id=0, sos_id=1):
    words = []
    for i in ids:
        if i == eos_id:
            break
        if i in (pad_id, sos_id):
            continue
        words.append(id2word.get(i, "<UNK>"))
    return " ".join(words)


# ===================== COLLATE FUNCTION =====================
def collate_fn(batch):
    src = [torch.tensor(x["input_ids"]) for x in batch]
    tgt = [torch.tensor(x["labels"]) for x in batch]

    src = torch.stack(src)
    tgt = torch.stack(tgt)

    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    return src, tgt_in, tgt_out


# ===================== MAIN TRAIN FUNCTION =====================
def run_transformer_config(config, config_id, run_id, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # ===================== LOAD TRANSFORMER DATA =====================
    train_ds = Dataset.from_list(load_split("train_transformer")).shuffle(seed=seed)
    val_ds   = Dataset.from_list(load_split("val_transformer")).shuffle(seed=seed)
    test_ds  = Dataset.from_list(load_split("test_transformer")).shuffle(seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # ===================== LOAD VOCAB (TRUTH SOURCE) =====================
    src_vocab = load_vocab("artifacts/vocab/src_vocab.json")
    tgt_vocab = load_vocab("artifacts/vocab/tgt_vocab.json")

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    id2tgt = invert_vocab(tgt_vocab)

    print(f"[INFO] SRC vocab size = {src_vocab_size}")
    print(f"[INFO] TGT vocab size = {tgt_vocab_size}")

    # ===================== INIT MODEL =====================
    model = TransformerTranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ===================== MLFLOW =====================
    mlflow.set_experiment("MLOps-NMT-EN-VI")
    run_name = f"transformer_config{config_id}_run{run_id}"

    # ===================== WANDB =====================
    wandb.init(
        project="MLOps-NMT-EN-VI",
        name=run_name,
        config=config,
        reinit=True
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "lr": config["lr"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "d_model": config["d_model"],
            "nhead": config["nhead"],
            "num_layers": config["num_layers"],
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "device": device
        })

        # ===================== TRAIN LOOP =====================
        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0.0

            for src, tgt_in, tgt_out in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                src, tgt_in, tgt_out = (
                    src.to(device),
                    tgt_in.to(device),
                    tgt_out.to(device)
                )

                optimizer.zero_grad()
                logits = model(src, tgt_in)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            wandb.log({"train_loss": avg_loss, "epoch": epoch})

        # ===================== BLEU EVALUATION =====================
        model.eval()
        preds, refs = [], []

        SOS_ID = tgt_vocab["<SOS>"]
        EOS_ID = tgt_vocab["<EOS>"]

        with torch.no_grad():
            for item in tqdm(test_ds, desc="BLEU (greedy decode)"):
                src = torch.tensor(item["input_ids"]).unsqueeze(0).to(device)
                ys = torch.full((1, 1), SOS_ID, dtype=torch.long).to(device)

                for _ in range(MAX_LEN):
                    out = model(src, ys)
                    next_token = out[:, -1].argmax(-1).unsqueeze(1)
                    ys = torch.cat([ys, next_token], dim=1)

                    if next_token.item() == EOS_ID:
                        break

                pred_sentence = decode_ids(
                    ys.squeeze().tolist(),
                    id2tgt,
                    eos_id=EOS_ID
                )

                ref_sentence = decode_ids(
                    item["labels"],
                    id2tgt,
                    eos_id=EOS_ID
                )

                preds.append(pred_sentence)
                refs.append([ref_sentence])

        bleu = corpus_bleu(preds, refs).score
        mlflow.log_metric("BLEU", bleu)
        wandb.log({"BLEU": bleu})

        # ===================== SAVE MODEL =====================
        artifact_dir = f"artifacts/transformer_config{config_id}"
        os.makedirs(artifact_dir, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    **config,
                    "src_vocab_size": src_vocab_size,
                    "tgt_vocab_size": tgt_vocab_size
                }
            },
            os.path.join(artifact_dir, "transformer.pt")
        )

        mlflow.log_artifacts(artifact_dir, artifact_path="model")
        wandb.finish()

    return bleu, avg_loss, avg_loss

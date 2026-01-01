import json
import os
import torch
import mlflow
import numpy as np
import wandb

from datasets import Dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from model.load_model import load_model
from training.config import MAX_LEN


# ===================== LOAD DATA =====================
def load_split(name):
    with open(f"data/processed/{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ===================== MAIN RUN =====================
def run_opus_config(config, config_id, run_id, seed):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ---------- LOAD DATASET ----------
    train = Dataset.from_list(load_split("train"))
    val   = Dataset.from_list(load_split("val"))
    test  = Dataset.from_list(load_split("test"))

    # ---------- REDUCE DATA (DEMO) ----------
    train = train.shuffle(seed=seed).select(range(200000)) 
    val = val.shuffle(seed=seed).select(range(2000)) 
    test = test.shuffle(seed=seed).select(range(2000))

    # ---------- LOAD MODEL (OPUS-MT) ----------
    tokenizer, model = load_model()  # Helsinki-NLP/opus-mt-en-vi

    # ---------- TOKENIZE ----------
    def tokenize(batch):
        return tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

    train = train.map(tokenize, batched=True)
    val   = val.map(tokenize, batched=True)
    test  = test.map(tokenize, batched=True)

    # ===================== WANDB =====================
    wandb.init(
        project="MLOps-NMT-EN-VI",
        name=f"config{config_id}_run{run_id}",
        config={
            "lr": config["lr"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "seed": seed,
            "model": "Helsinki-NLP/opus-mt-en-vi"
        },
        reinit=True
    )

    # ===================== TRAINING ARGUMENTS =====================
    train_args = TrainingArguments(
        output_dir="tmp/trainer",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        logging_strategy="epoch",
        save_strategy="no",
        seed=seed,
        report_to="wandb",
        fp16=torch.cuda.is_available()
    )

    # ===================== MLFLOW =====================
    mlflow.set_experiment("MLOps-NMT-EN-VI")

    with mlflow.start_run(run_name=f"config{config_id}"):

        # ---------- LOG PARAMS ----------
        mlflow.log_params(config)
        mlflow.log_param("seed", seed)
        mlflow.log_param("dataset_version", "processed_v1")
        mlflow.log_param("model_name", model.config._name_or_path)
        mlflow.log_param(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===================== TRAINER =====================
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train,
            eval_dataset=val,
            tokenizer=tokenizer
        )

        # ---------- TRAIN ----------
        train_output = trainer.train()
        train_loss = train_output.training_loss
        mlflow.log_metric("train_loss", train_loss)

        # ---------- VALIDATION ----------
        eval_metrics = trainer.evaluate()
        val_loss = eval_metrics["eval_loss"]
        mlflow.log_metric("val_loss", val_loss)

        # ===================== BLEU (CPU) =====================
        print(">>> STARTING CPU GENERATION FOR BLEU <<<")

        model_cpu = model.to("cpu").eval()

        pred_texts = []
        label_texts = []

        for i in tqdm(range(len(test)), desc="Generating (CPU)"):
            item = test[i]

            input_ids = torch.tensor(item["input_ids"]).unsqueeze(0)
            attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0)

            with torch.no_grad():
                gen_ids = model_cpu.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_LEN
                )

            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            label_ids = np.where(
                np.array(item["labels"]) != -100,
                item["labels"],
                tokenizer.pad_token_id
            )
            label = tokenizer.decode(label_ids, skip_special_tokens=True)

            pred_texts.append(pred)
            label_texts.append(label)

        refs = [[x] for x in label_texts]
        bleu = corpus_bleu(pred_texts, refs).score

        mlflow.log_metric("BLEU", bleu)
        wandb.log({
            "BLEU": bleu,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f">>> BLEU = {bleu:.2f} <<<")

        # ===================== SAVE MODEL =====================
        artifact_dir = f"artifacts/opus_config{config_id}"
        os.makedirs(artifact_dir, exist_ok=True)

        model_cpu.save_pretrained(artifact_dir)
        tokenizer.save_pretrained(artifact_dir)

        mlflow.log_artifacts(artifact_dir, artifact_path="model")
        mlflow.log_param("model_saved", True)
        mlflow.log_param("model_save_type", "huggingface_save_pretrained")

        wandb.finish()

    return bleu, train_loss, val_loss

import json
import os
import torch
import mlflow
import numpy as np

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
def run_config(config, config_id, run_id, seed):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ---------- LOAD DATASET ----------
    train = Dataset.from_list(load_split("train"))
    val   = Dataset.from_list(load_split("val"))
    test  = Dataset.from_list(load_split("test"))

    # ---------- REDUCE DATA (NHANH, ĐỦ DEMO) ----------
    train = train.shuffle(seed=seed).select(range(200000))
    val   = val.shuffle(seed=seed).select(range(2000))
    test  = test.shuffle(seed=seed).select(range(2000))   # 100 mẫu cho BLEU
    # ---------- LOAD MODEL (OPUS-MT) ----------
    tokenizer, model = load_model()   # MODEL_NAME = Helsinki-NLP/opus-mt-en-vi

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

    # ===================== TRAINING ARGUMENTS (GPU) =====================
    train_args = TrainingArguments(
        output_dir=f"models/c{config_id}_r{run_id}",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        logging_strategy="epoch",
        save_strategy="no",
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    # ===================== MLFLOW =====================
    mlflow.set_experiment("MLOps-NMT-EN-VI")

    with mlflow.start_run(run_name=f"config{config_id}_run{run_id}"):

        # ---------- LOG CONTEXT ----------
        mlflow.log_params(config)
        mlflow.log_param("seed", seed)
        mlflow.log_param("dataset_version", "processed_v1")
        mlflow.log_param("model_name", model.config._name_or_path)
        mlflow.log_param(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===================== TRAINER (GPU TRAIN) =====================
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

        # ==========================================================
        # ================= CPU GENERATE + BLEU ====================
        # ==========================================================

        print(">>> STARTING CPU GENERATION FOR BLEU <<<")

        model_cpu = model.to("cpu")
        model_cpu.eval()

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

            # decode prediction
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # decode label (xử lý -100)
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

        print(f">>> BLEU = {bleu:.2f} <<<")

        # ===================== SAVE & LOG MODEL AS MLFLOW ARTIFACT =====================
        artifact_dir = f"artifacts/config{config_id}_run{run_id}"
        os.makedirs(artifact_dir, exist_ok=True)

        # LƯU MODEL + TOKENIZER THEO CHUẨN HUGGINGFACE
        model_cpu.save_pretrained(artifact_dir)
        tokenizer.save_pretrained(artifact_dir)

        # LOG LÊN MLFLOW DƯỚI DẠNG ARTIFACT
        mlflow.log_artifacts(artifact_dir, artifact_path="model")

        # GHI NHẬN LÀ MODEL ĐÃ ĐƯỢC LƯU
        mlflow.log_param("model_saved", True)
        mlflow.log_param("model_save_type", "huggingface_save_pretrained")

    return bleu, train_loss, val_loss

from training.run_opus import run_opus_config
from training.run_transformer import run_transformer_config

# ===================== HYPERPARAMETER CONFIGS =====================
HYPERPARAMETER_CONFIGS = [
    # ================= OPUS-MT (PRETRAINED) =================
    {
        "model_type": "opus",
        "lr": 2e-4,
        "batch_size": 16,
        "epochs": 3
    },
    {
        "model_type": "opus",
        "lr": 5e-4,
        "batch_size": 8,
        "epochs": 1
    },

    # ================= TRANSFORMER (FROM SCRATCH) =================
    {
        "model_type": "transformer",
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 5,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "src_vocab_size": 32000,
        "tgt_vocab_size": 32000
    },
    {
        "model_type": "transformer",
        "lr": 5e-5,
        "batch_size": 16,
        "epochs": 8,
        "d_model": 512,
        "nhead": 8,
        "num_layers": 4,
        "src_vocab_size": 32000,
        "tgt_vocab_size": 32000
    }
]

SEED = 42
RUN_ID = 1

# ===================== MAIN =====================
if __name__ == "__main__":
    for idx, config in enumerate(HYPERPARAMETER_CONFIGS, start=1):
        print("\n===================================")
        print(f"===== RUN CONFIG {idx} =====")
        print(f"Model type: {config['model_type']}")
        print(config)
        print("===================================")

        # ---------- OPUS-MT ----------
        if config["model_type"] == "opus":
            bleu, train_loss, val_loss = run_opus_config(
                config=config,
                config_id=idx,
                run_id=RUN_ID,
                seed=SEED
            )

        # ---------- TRANSFORMER (SCRATCH) ----------
        elif config["model_type"] == "transformer":
            bleu, train_loss, val_loss = run_transformer_config(
                config=config,
                config_id=idx,
                run_id=RUN_ID,
                seed=SEED
            )

        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")

        print("\n[RESULT]")
        print(f"BLEU       : {round(bleu, 2)}")
        print(f"Train loss : {round(train_loss, 4)}")
        print(f"Val loss   : {round(val_loss, 4)}")

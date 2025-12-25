from training.run_config import run_config

HYPERPARAMETER_CONFIGS = [
    {"lr": 5e-4, "batch_size": 8,  "epochs": 1},
    {"lr": 2e-4, "batch_size": 16,  "epochs": 3},
    {"lr": 1e-4, "batch_size": 16, "epochs": 2}
]

SEED = 42

if __name__ == "__main__":
    for idx, config in enumerate(HYPERPARAMETER_CONFIGS, start=1):
        print(f"\n===== RUN CONFIG {idx} =====")
        print(config)

        bleu, train_loss, val_loss = run_config(
            config=config,
            config_id=idx,
            run_id=1,
            seed=SEED
        )

        print(f"[RESULT] Config {idx}")
        print("BLEU:", round(bleu, 2))
        print("Train loss:", round(train_loss, 4))
        print("Val loss:", round(val_loss, 4))

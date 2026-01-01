# mlops-neural-machine-translation
dvc repro
mlflow ui --port 5000
uvicorn serving.app:app --reload --port 8000

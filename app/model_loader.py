from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import joblib
import os
import json

SUBSCRIPTION_ID = "297a29d5-77dd-42d6-bd17-a8e3ce2429fe"
RESOURCE_GROUP = "rg-mlops-demo"
WORKSPACE_NAME = "mlops-workspace"

CONFIG_PATH = "model_config.json"
LOCAL_MODEL_DIR = "downloaded_model"


def load_model_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_model():
    config = load_model_config()

    model_name = config["model_name"]
    model_version = str(config["model_version"])

    local_model_path = f"{LOCAL_MODEL_DIR}/{model_name}/model.pkl"

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    if not os.path.exists(local_model_path):
        ml_client = MLClient(
            DefaultAzureCredential(),
            SUBSCRIPTION_ID,
            RESOURCE_GROUP,
            WORKSPACE_NAME
        )

        ml_client.models.download(
            name=model_name,
            version=model_version,
            download_path=LOCAL_MODEL_DIR
        )

    return joblib.load(local_model_path)
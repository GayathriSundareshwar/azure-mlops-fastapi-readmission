from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment, Model
from azure.identity import DefaultAzureCredential
import json

subscription_id = "297a29d5-77dd-42d6-bd17-a8e3ce2429fe"
resource_group = "rg-mlops-demo"
workspace_name = "mlops-workspace"

DATA_NAME = "readmission_data"
DATA_VERSION = "1"
COMPUTE_NAME = "cpu-cluster"
MODEL_NAME = "readmission-model"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)

data_asset = ml_client.data.get(
    name=DATA_NAME,
    version=DATA_VERSION
)

custom_env = Environment(
    name="readmission-training-env",
    description="Training environment with fixed sklearn version",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file={
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "pandas",
                    "numpy==2.0.2",
                    "scikit-learn==1.5.2",
                    "joblib",
                    "mlflow",
                    "azureml-mlflow"
                ]
            }
        ]
    }
)

job = command(
    code="./training",
    command="python train_azure.py --data ${{inputs.data}} --model_output ${{outputs.model_output}}",
    inputs={
        "data": Input(
            type="uri_file",
            path=data_asset.path
        )
    },
    outputs={
        "model_output": Output(type="uri_folder")
    },
    environment=custom_env,
    compute=COMPUTE_NAME,
    experiment_name="readmission-training",
    display_name="readmission-model-training-auto"
)

returned_job = ml_client.jobs.create_or_update(job)

print("Submitted Azure ML training job:")
print(returned_job.name)

print("Waiting for training job to complete...")
ml_client.jobs.stream(returned_job.name)

completed_job = ml_client.jobs.get(returned_job.name)

if completed_job.status != "Completed":
    raise RuntimeError(f"Training job failed with status: {completed_job.status}")

model_path = f"azureml://jobs/{returned_job.name}/outputs/model_output/paths/model.pkl"

model = Model(
    path=model_path,
    name=MODEL_NAME,
    description="Hospital readmission prediction model",
    type="custom_model"
)

registered_model = ml_client.models.create_or_update(model)

print("Model registered:")
print(registered_model.name)
print(registered_model.version)

config = {
    "model_name": registered_model.name,
    "model_version": registered_model.version
}

with open("model_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Saved model_config.json")
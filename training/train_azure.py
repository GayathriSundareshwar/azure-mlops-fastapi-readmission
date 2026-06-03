import argparse
import os
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True
)

parser.add_argument(
    "--model_output",
    type=str,
    required=True
)

args = parser.parse_args()


# Load data from Azure ML input
df = pd.read_csv(args.data)

# Split features and target
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Log metrics to Azure ML / MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")

# Save model to Azure ML output folder
os.makedirs(args.model_output, exist_ok=True)

model_path = os.path.join(
    args.model_output,
    "model.pkl"
)

joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
print("Training completed successfully")
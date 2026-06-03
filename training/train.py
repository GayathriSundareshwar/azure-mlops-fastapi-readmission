from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib
import json

# 1. Create dataset
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42,
)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 3. Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)

# 5. Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm.tolist(),
}

# 6. Print metrics
print("Model Metrics")
print("-------------")
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")
print("\nConfusion Matrix:")
print(cm)

# 7. Save model
joblib.dump(model, "artifacts/model.pkl")
print("\nSaved model.pkl")

# 8. Save metrics
with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved metrics.json")

import numpy as np
import joblib
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")

# Load models
dnn_model = load_model("models/dnn_model.keras")
rf_model = joblib.load("models/random_forest.pkl")

# DNN predictions
dnn_pred = np.argmax(dnn_model.predict(X_test, verbose=0), axis=1)

# Random Forest predictions
rf_pred = rf_model.predict(X_test)

# Calculate metrics
results = []

for name, predictions in [
    ("DNN", dnn_pred),
    ("Random Forest", rf_pred)
]:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="macro"),
        "Recall": recall_score(y_test, predictions, average="macro"),
        "F1 Score": f1_score(y_test, predictions, average="macro")
    })

# Display comparison
comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv("model_comparison.csv", index=False)

print("\nComparison saved successfully!")

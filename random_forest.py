
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load preprocessed data
X_train = np.load("models/X_train.npy")
X_test = np.load("models/X_test.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")

label_encoder = joblib.load("models/label_encoder.pkl")
class_names = label_encoder.classes_

# 2. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

# 3. Train model
rf_model.fit(X_train, y_train)

# 4. Make predictions
y_pred = rf_model.predict(X_test)

# 5. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Evaluation")
print("Test Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Greens", values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.show()

# 7. Save model
joblib.dump(rf_model, "models/random_forest.pkl")

print("\nRandom Forest model saved successfully!")

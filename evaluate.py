
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load saved model and test data
model = load_model("models/dnn_model.keras")
X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")

label_encoder = joblib.load("models/label_encoder.pkl")
class_names = label_encoder.classes_

# Predict test data
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", values_format="d")
plt.title("DNN Confusion Matrix")
plt.show()

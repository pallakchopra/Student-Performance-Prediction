
import numpy as np
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load preprocessed data
X_train = np.load("models/X_train.npy")
X_test = np.load("models/X_test.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")

# 3. Build DNN model
model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(4, activation="softmax")
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Display model architecture
model.summary()

# 6. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# 7. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 8. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nDNN Model Evaluation")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# 9. Save trained model
os.makedirs("models", exist_ok=True)
model.save("models/dnn_model.keras")

print("\nModel saved successfully!")

# 10. Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("DNN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 11. Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("DNN Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


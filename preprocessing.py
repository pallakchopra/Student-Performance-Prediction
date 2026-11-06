
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("student-por.csv", sep=";")

# 2. Convert final grades into performance categories
def categorize_grade(grade):
    if grade <= 9:
        return "Low"
    elif grade <= 13:
        return "Average"
    elif grade <= 16:
        return "Good"
    else:
        return "Excellent"

df["Performance"] = df["G3"].apply(categorize_grade)

# 3. Separate input features and target
X = df.drop(columns=["G3", "Performance"])
y = df["Performance"]

# 4. Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 5. Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# 6. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# 7. Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 8. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Fit preprocessing only on training data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# 10. Create models folder
os.makedirs("models", exist_ok=True)

# 11. Save preprocessing objects and datasets
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

np.save("models/X_train.npy", X_train)
np.save("models/X_test.npy", X_test)
np.save("models/y_train.npy", y_train)
np.save("models/y_test.npy", y_test)

# 12. Display results
print("Preprocessing completed successfully!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of input features after encoding:", X_train.shape[1])

print("\nPerformance Categories:")
print(label_encoder.classes_)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

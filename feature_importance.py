
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained Random Forest and preprocessor
model = joblib.load("models/random_forest.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Get importance scores
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 7))

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Save results
importance_df.to_csv("feature_importance.csv", index=False)

print("\nFeature importance saved successfully!")

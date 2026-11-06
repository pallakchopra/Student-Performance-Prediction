
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student-por.csv", sep=";")

# 1. Statistical summary
print("Statistical Summary:")
print(df.describe())

# 2. Distribution of final grades
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.show()

# 3. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.show()

# 4. Previous grades vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(x="G2", y="G3", data=df)
plt.title("Second Period Grade vs Final Grade")
plt.xlabel("G2")
plt.ylabel("G3")
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.show()


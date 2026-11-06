import pandas as pd

# Load the dataset
df = pd.read_csv("student-por.csv", sep=";")

# Display first 5 rows
print(df.head())

# Display dataset shape
print("Dataset Shape:", df.shape)

# Display column names
print("Columns:", df.columns.tolist())

# Check missing values
print(df.isnull().sum())

# Display dataset information
print(df.info())

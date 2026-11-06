
# 🎓 Student Performance Prediction System

## 📌 Project Overview

The Student Performance Prediction System is a Machine Learning-based web application that predicts students' academic performance based on their demographic, family, social, and academic attributes.

The system classifies students into four performance categories:
- Low
- Average
- Good
- Excellent

It uses two Machine Learning approaches, Deep Neural Network (DNN) and Random Forest, to compare predictive performance.

The application also provides feature importance analysis to identify the attributes that contribute most to predictions.

## 🚀 Features

- Student performance prediction through an interactive web interface.
- Classification into four performance categories.
- Deep Neural Network and Random Forest model comparison.
- Feature importance visualization.
- Prediction confidence and performance category probabilities.
- Interactive charts and a user-friendly dashboard.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly
- Matplotlib and Seaborn
- Joblib

## 📂 Dataset

The project uses the Student Performance Dataset, specifically the Portuguese language dataset (`student-por.csv`).

It contains 649 student records and 33 original attributes covering:
- Student demographics
- Family background
- Study habits
- Social activities
- Academic performance

The final grade (G3) is converted into four performance categories.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Created performance categories using the final grade (G3).
3. Separated input features and target variable.
4. Encoded target labels using LabelEncoder.
5. Applied StandardScaler to numerical features.
6. Applied OneHotEncoder to categorical features.
7. Split the dataset into 80% training and 20% testing data.
8. Saved the preprocessing objects and processed datasets.

## 🤖 Machine Learning Models

### 1. Deep Neural Network (DNN)

A neural network trained using TensorFlow/Keras to classify student performance into four categories.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to predict student performance.

## 📊 Model Evaluation

Both models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

### Model Comparison Results

| Model | Accuracy |
|---|---:|
| Deep Neural Network | 76.92% |
| Random Forest | 80.77% |

Random Forest achieved higher test accuracy on the current test split.

## 🎯 Feature Importance

Feature importance analysis was performed using the Random Forest model to identify the most influential input features.

The top features include:
- Second Period Grade (G2)
- First Period Grade (G1)
- Past Class Failures
- Student Age
- School Absences
- Mother's Education Level
- Going Out with Friends
- Father's Education Level
- Weekly Study Time
- Weekend Alcohol Consumption

## 💻 Application Pages

### Predictor
Allows users to enter student details and receive a predicted performance category with prediction confidence and category probabilities.

### Model Comparison
Displays a comparison of DNN and Random Forest using evaluation metrics.

### Feature Importance
Visualizes the top 10 important student attributes.

## 📁 Project Structure

```text
Student_Performance_Prediction/
│
├── app.py
├── data_analysis.py
├── eda.py
├── preprocessing.py
├── train.py
├── random_forest.py
├── evaluate.py
├── compare_models.py
├── feature_importance.py
│
├── student-por.csv
├── model_comparison.csv
├── feature_importance.csv
│
└── models/
    ├── dnn_model.keras
    ├── random_forest.pkl
    ├── preprocessor.pkl
    ├── label_encoder.pkl
    ├── X_train.npy
    ├── X_test.npy
    ├── y_train.npy
    └── y_test.npy
```

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Navigate to the project folder

```bash
cd Student_Performance_Prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow streamlit plotly matplotlib seaborn joblib
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Scope

- Improve model performance through hyperparameter tuning.
- Experiment with additional classification algorithms.
- Add more interactive student performance insights.
- Explore personalized academic improvement recommendations.

## 👩‍💻 Project Information

**Project:** Student Performance Prediction System

**Domain:** Machine Learning and Artificial Intelligence

**Language:** Python

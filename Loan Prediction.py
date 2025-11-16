# ---------------------------------------------
# LOAN APPROVAL PREDICTION - FULL ML PIPELINE
# ---------------------------------------------

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# STEP 2: Load Dataset (LOCAL WINDOWS PATH)
df = pd.read_csv(r"C:\Users\welcome\Desktop\Loan Prediction Analysis Model\loan_prediction.csv")
# If the file is in same folder as script, use:
# df = pd.read_csv("loan_prediction.csv")

# STEP 3: Preview Data
print("Data Preview:\n")
print(df.head())

print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

# ---------------------------------------------
# DATA CLEANING
# ---------------------------------------------

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include='object').columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Fill categorical missing values with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numeric missing values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# ---------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------

# Loan Status Count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Loan_Status')
plt.title("Loan Approval Distribution")
plt.show()

# Gender vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender', hue='Loan_Status')
plt.title("Loan Status by Gender")
plt.show()

# Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['ApplicantIncome'], kde=True)
plt.title("Applicant Income Distribution")
plt.show()

# ---------------------------------------------
# LABEL ENCODING
# ---------------------------------------------
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ---------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# ---------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# MODEL TRAINING - Random Forest
# ---------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------
# PREDICTION & EVALUATION
# ---------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ---------------------------------------------
# FINAL PREDICTION FUNCTION
# ---------------------------------------------
def predict_loan(input_data):
    df_input = pd.DataFrame([input_data])

    # Fill missing values
    for col in df_input.columns:
        if col in categorical_cols:
            df_input[col] = df_input[col].fillna(df[col].mode()[0])
        else:
            df_input[col] = df_input[col].fillna(df[col].median())

    # Label encode
    for col in df_input.columns:
        if col in categorical_cols:
            df_input[col] = le.fit_transform(df_input[col])

    return model.predict(df_input)[0]

# Example prediction
sample = X.iloc[0].to_dict()
print("\nSample Prediction (0 = Rejected, 1 = Approved):", predict_loan(sample))

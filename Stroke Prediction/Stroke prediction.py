# Basic libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Handle imbalance
from imblearn.over_sampling import SMOTE

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Basic info
print(df.shape)
print(df.head())
print(df.info())

# Target distribution
sns.countplot(x='stroke', data=df)
plt.title("Stroke Distribution")
plt.show()

# Check missing values
print(df.isnull().sum())

# Age vs Stroke
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="age", hue="stroke", bins=30, kde=True)
plt.title("Age Distribution by Stroke")
plt.show()

# Glucose vs Stroke
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="avg_glucose_level", hue="stroke", bins=30, kde=True)
plt.title("Glucose Level by Stroke")
plt.show()


# Drop 'id' column
df = df.drop("id", axis=1)

# Fill missing BMI with mean
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Before SMOTE:", y.value_counts())
print("After SMOTE:", y_res.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("Logistic Regression Report")
print(classification_report(y_test, y_pred_log))

rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report")
print(classification_report(y_test, y_pred_rf))

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Report")
print(classification_report(y_test, y_pred_xgb))

models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

plt.figure(figsize=(7,6))
for name, model in models.items():
    y_pred_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Comparison")
plt.legend()
plt.show()

importances = rf_model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)[:10]

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Top 10 Important Features for Stroke Prediction")
plt.show()




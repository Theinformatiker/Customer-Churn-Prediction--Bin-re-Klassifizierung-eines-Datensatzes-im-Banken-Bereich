# Source: https://www.kaggle.com/code/saumyagupta2025/ann-for-customer-churn-prediction-87-accuracy

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
import time
# Load the CSV file
df = pd.read_csv('Churn_Modelling2.CSV')

# Data preprocessing
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# One-Hot-Encoding for 'Geography' and 'Gender'
df = pd.get_dummies(df, columns=['Geography', 'Gender'])

# Feature Engineering: Create a new feature
df['Age_Balance'] = df['Age'] * df['Balance']

# Splitting features and target
X = df.drop('Exited', axis=1)
y = df['Exited']


# Startzeit erfassen
start_time = time.time()

# Handling class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Outlier detection with Z-score
numeric_cols = X_res.select_dtypes(include=[np.number]).columns
z_scores = stats.zscore(X_res[numeric_cols])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X_res = X_res[filtered_entries]
y_res = y_res[filtered_entries]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Scaling numerical features
numeric_features = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

# Creating copies to avoid SettingWithCopyWarning
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Defining classifiers
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
lgbm = LGBMClassifier(random_state=42)
catboost = CatBoostClassifier(random_state=42, verbose=0)

# Define parameter space for Bayesian Optimization
param_space_xgb = {
    'n_estimators': Integer(500, 2000),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.05, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0)
}

param_space_lgbm = {
    'n_estimators': Integer(500, 2000),
    'max_depth': Integer(10, 30),
    'learning_rate': Real(0.05, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'min_gain_to_split': Real(0.0, 0.1),
    'min_data_in_leaf': Integer(1, 20)
}

param_space_catboost = {
    'iterations': Integer(500, 2000),
    'depth': Integer(3, 15),
    'learning_rate': Real(0.05, 0.3, prior='log-uniform'),
    'l2_leaf_reg': Real(1, 9)
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Bayesian Search for XGBoost
bayes_search_xgb = BayesSearchCV(
    estimator=xgb,
    search_spaces=param_space_xgb,
    n_iter=26,
    scoring='roc_auc',
    cv=skf,
    n_jobs=-1,
    random_state=42
)

# Train XGBoost model with Bayesian optimization
bayes_search_xgb.fit(X_train_scaled, y_train)
best_xgb = bayes_search_xgb.best_estimator_

# Bayesian Search for LightGBM
bayes_search_lgbm = BayesSearchCV(
    estimator=lgbm,
    search_spaces=param_space_lgbm,
    n_iter=26,
    scoring='roc_auc',
    cv=skf,
    n_jobs=-1,
    random_state=42
)

# Train LightGBM model with Bayesian optimization
bayes_search_lgbm.fit(X_train_scaled, y_train)
best_lgbm = bayes_search_lgbm.best_estimator_

# Bayesian Search for CatBoost
bayes_search_catboost = BayesSearchCV(
    estimator=catboost,
    search_spaces=param_space_catboost,
    n_iter=26,
    scoring='roc_auc',
    cv=skf,
    n_jobs=-1,
    random_state=42
)

# Train CatBoost model with Bayesian optimization
bayes_search_catboost.fit(X_train_scaled, y_train)
best_catboost = bayes_search_catboost.best_estimator_

# Create an ensemble with VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('catboost', best_catboost)
    ],
    voting='soft',
    n_jobs=-1
)

# Train the ensemble model
ensemble.fit(X_train_scaled, y_train)

# Predictions and evaluation on the test set
y_pred = ensemble.predict(X_test_scaled)
y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

# Classification report
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)

print(f"Ensemble Model Classification Report:\n{report}")
print(f"ROC AUC Score: {roc_auc}")
print(f"Accuracy: {accuracy}")

# Endzeit erfassen
end_time = time.time()

# Berechnung der verstrichenen Zeit
elapsed_time = end_time - start_time
print(f"Verstrichene Zeit: {elapsed_time:.4f} Sekunden")

# Berechnung der Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Darstellung der Konfusionsmatrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix of Ensemble Model")
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()




fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()



importances = best_xgb.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_idx], align="center")
plt.xticks(range(len(importances)), features[sorted_idx], rotation=90)
plt.title("Feature Importance from XGBoost")
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(y_proba[y_test == 0], bins=25, alpha=0.6, label='Class 0', color='blue')
plt.hist(y_proba[y_test == 1], bins=25, alpha=0.6, label='Class 1', color='red')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution of Predictions')
plt.legend()
plt.show()


train_sizes, train_scores, test_scores = learning_curve(
    ensemble, X_train_scaled, y_train, cv=skf, n_jobs=-1, scoring='accuracy')
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
plt.plot(train_sizes, test_mean, label='Validation Score', color='orange')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', color='black', label='Calibration Curve')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()

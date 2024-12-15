import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import multiprocessing
import numpy as np
import shap  # SHAP library

# LOGICAL CPU CORES
num_cores = multiprocessing.cpu_count()
os.environ['LOKY_MAX_CPU_COUNT'] = str(max(1, num_cores // 2))
print(f"Using {os.environ['LOKY_MAX_CPU_COUNT']} out of {num_cores} logical CPU cores.")

# LOAD THE DATASET
file_path = 'C:/Users/amano/PycharmProjects/pythonProject/final/dataset/preprocessed_player_stats.csv'
player_stats = pd.read_csv(file_path)

# HANDLE MISSING VALUES
numeric_cols = player_stats.select_dtypes(include=['number']).columns
categorical_cols = player_stats.select_dtypes(include=['object']).columns
numeric_imputer = SimpleImputer(strategy='median')
player_stats[numeric_cols] = numeric_imputer.fit_transform(player_stats[numeric_cols])
categorical_imputer = SimpleImputer(strategy='most_frequent')
for col in categorical_cols:
    player_stats[col] = categorical_imputer.fit_transform(player_stats[[col]])

# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in ['Position', 'Team', 'League']:
    player_stats[col] = label_encoder.fit_transform(player_stats[col])

# REMOVE LOW-IMPORTANCE FEATURES
low_importance_features = ['Corners', 'RedCards', 'Errors', 'ID']
player_stats = player_stats.drop(columns=low_importance_features)

# TARGET VALUE
X = player_stats.drop(columns=['Injured'])
y = player_stats['Injured']

# SPLIT THE DATA (TRAIN AND TEST)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALE THE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# HANDLE CLASS IMBALANCE USING SMOTETOMEK
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

# BEST HYPERPARAMETERS
best_rf_model = RandomForestClassifier(
    random_state=42,
    bootstrap=True,
    class_weight='balanced',
    max_depth=50,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=700
)

# TRAIN THE MODEL
best_rf_model.fit(X_train_resampled, y_train_resampled)
y_pred = best_rf_model.predict(X_test_scaled)

import joblib
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")

print("Features used during model training:", X.columns.tolist())

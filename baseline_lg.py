import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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
label_encoder = LabelEncoder()
for col in ['Position', 'Team', 'League']:
    player_stats[col] = label_encoder.fit_transform(player_stats[col])

# TARGET VALUE
X = player_stats.drop(columns=['Injured'])
y = player_stats['Injured']

# SPLIT THE DATA (TRAIN AND TEST)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALE THE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# INITIALIZE LOGISTIC REGRESSION
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train)
y_pred = log_reg_model.predict(X_test_scaled)

# EVALUATE THE MODEL
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# PRINT PERFORMANCE METRICS
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

# CLASSIFICATION REPORT
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# CONFUSIONS MATRIX
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# PREDICT PROBABILITIES FOR ROC-AUC
y_pred_prob = log_reg_model.predict_proba(X_test_scaled)[:, 1]

# CALCULATE ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC-AUC Score: {roc_auc:.2f}")

# PLOT ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

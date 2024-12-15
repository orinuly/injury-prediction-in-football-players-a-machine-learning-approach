import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import json


file_path = 'cleaned_player_stats_with_renamed_columns.csv'
player_stats = pd.read_csv(file_path)

# MISSING VALUES
numeric_cols = player_stats.select_dtypes(include=['number']).columns
categorical_cols = player_stats.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='median')
player_stats[numeric_cols] = numeric_imputer.fit_transform(player_stats[numeric_cols])
categorical_imputer = SimpleImputer(strategy='most_frequent')
player_stats[categorical_cols] = categorical_imputer.fit_transform(player_stats[categorical_cols])

# LABEL ENCODING
label_encoder = LabelEncoder()
label_mappings = {}

for col in ['Position', 'Team', 'League']:
    player_stats[col] = label_encoder.fit_transform(player_stats[col])
    label_mappings[col] = {str(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}


for col, mapping in label_mappings.items():
    print(f"Mapping for {col}:")
    print(mapping)

# SAVE
preprocessed_file_path = 'preprocessed_player_stats.csv'
player_stats.to_csv(preprocessed_file_path, index=False)

with open('label_mappings.json', 'w') as f:
    json.dump(label_mappings, f)

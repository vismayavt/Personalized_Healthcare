import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv(r"C:\Users\visma\OneDrive\Desktop\Personalized_Healthcare_Recommendation\healthcare_dataset.csv")  # Replace with your actual CSV file name


# Drop rows with missing values
data.dropna(inplace=True)

# Target and features
target = "Medical Condition"
X = data.drop(columns=[target])
y = data[target]

# Feature engineering
X["health_index"] = X["Age"] / (X["Billing Amount"] + 1)

numeric_features = ["Age", "Billing Amount", "Room Number", "health_index"]
categorical_features = ["Gender", "Blood Type", "Admission Type", "Insurance Provider"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save the full pipeline
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ pipeline.pkl saved successfully.")


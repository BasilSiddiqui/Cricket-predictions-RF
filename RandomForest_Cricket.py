import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score
import joblib

# Load data
df = pd.read_excel(r"C:\Users\basil\OneDrive\Desktop\Base\Other\Datasets\Cricket\Cricket stats.xlsx", sheet_name="Matches")

# Target and Features
y = df['Team']
X = df.drop(columns=['Team', 'Date'])  # keep 'Type' for one-hot encoding

# One-hot encoding categorical variables
X = pd.get_dummies(X)
feature_columns = X.columns  # Save feature columns for later

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --------- Predicting on New Data ---------
# Raw input (same columns as in original df before encoding)
new_data = pd.DataFrame([{
    'Overs': 4,
    'Maidens': 0,
    'Runs': 46,
    'Wickets': 2,
    'Type': 'B,C',
    'Balls': 24,
    'Economy': 11.5,
    'Strike rate': 12,
    'Average': 23,
    'Ground': 'Al Batayeh 2',
    'Opponent': 'VK 11',
    'Win/Loss': "W"
}])

# One-hot encode new_data
new_data_encoded = pd.get_dummies(new_data)

# Add missing columns
for col in feature_columns:
    if col not in new_data_encoded:
        new_data_encoded[col] = 0

# Ensure column order matches training data
new_data_encoded = new_data_encoded[feature_columns]

# Scale
new_data_scaled = scaler.transform(new_data_encoded)

# Predict
prediction = model.predict(new_data_scaled)
prob = model.predict_proba(new_data_scaled)

print("Predicted class:", prediction[0])
print("Probability of each class:", dict(zip(model.classes_, prob[0])))

qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
print("Quadratic Weighted Kappa (QWK):", qwk)


joblib.dump(model, 'random_forest_model.joblib') #Saving the model

loaded_model = joblib.load('random_forest_model.joblib') #Loading the model for later use
prediction = loaded_model.predict(X_test)
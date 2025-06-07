# 🏏 Cricket Match Predictor using Random Forest

This project is a simple demonstration of a supervised machine learning pipeline to predict the **team name** (`y`) based on match statistics like overs, wickets, economy, opponent, and ground, using the **Random Forest Classifier**. I've used it to predict teams, we could use it to predict pretty much any outcome in theory.

---

## 📁 Dataset

- **Source**: A personal Excel file (my own statistics from club cricket): `Cricket stats.xlsx`
- **Sheet**: `Matches`
- **Target variable**: `Team`
- **Features** include:
  - Overs, Maidens, Runs, Wickets
  - Balls, Economy, Strike Rate, Average
  - Ground, Opponent, Win/Loss
  - And others...

**Note:** While the dataset was small and not ideal for training a robust model, it was sufficient for demonstrating the **end-to-end workflow** of a classification model.

---

## 🧠 Objective

To predict the **team name** given match-level stats using Random Forest, a decision-tree based ensemble model.

---

## ⚙️ Steps in the ML Pipeline

### 1. **Load and Explore Data**
```python
df = pd.read_excel("Cricket stats.xlsx", sheet_name="Matches")
````

### 2. **Prepare Features and Labels**

```python
X = df.drop(columns=['Team', 'Date', 'Type'])  # Drop target and irrelevant columns
y = df['Team']
```

### 3. **One-Hot Encoding for Categorical Variables**

```python
X = pd.get_dummies(X)  # Encode categorical features
feature_columns = X.columns  # Save feature structure for prediction
```

### 4. **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. **Standardization (Feature Scaling)**

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. **Train Random Forest Classifier**

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 7. **Evaluate Model**

```python
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🧪 New Prediction Example

Here’s how we predict the team based on new match data:

```python
new_data = pd.DataFrame([{
    'Overs': 4,
    'Maidens': 0,
    'Runs': 46,
    'Wickets': 2,
    'Balls': 24,
    'Economy': 11.5,
    'Strike rate': 12,
    'Average': 23,
    'Ground': 'Al Batayeh 2',
    'Opponent': 'VK 11',
    'Win/Loss': "W"
}])

# One-hot encode and align columns
new_data = pd.get_dummies(new_data)
for col in feature_columns:
    if col not in new_data:
        new_data[col] = 0
new_data = new_data[feature_columns]

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)
```

---

## 🧾 Model Output

```
Confusion Matrix:
[[1 0 0 0 0 0 0]
 [0 0 1 0 0 0 0]
 [0 0 4 0 0 1 0]
 [0 0 0 1 0 1 0]
 [1 0 0 0 0 0 0]
 [0 0 0 0 0 6 0]
 [0 0 0 0 0 1 0]]

Classification Report:
              precision    recall  f1-score   support
12th Man Legends     0.50      1.00      0.67         1
Arab Unity           0.00      0.00      0.00         1
Defenders            0.80      0.80      0.80         5
Good Team            1.00      0.50      0.67         2
Gulf Cricket Club    0.00      0.00      0.00         1
Heriot Watt          0.67      1.00      0.80         6
Kricket Spero        0.00      0.00      0.00         1

Accuracy: 71%
Quadratic Weighted Kappa (QWK): 0.7077

Predicted class: Gulf Cricket Club
Probability of each class:
{
 '12th Man Legends': 0.12,
 'Arab Unity': 0.0,
 'Defenders': 0.08,
 'Good Team': 0.0,
 'Gulf Cricket Club': 0.48,
 'Heriot Watt': 0.26,
 'Kricket Spero': 0.06,
 'VK 11': 0.0
}
```

---

## 💡 Notes

* **'Type'** column was dropped due to inconsistent string values (e.g., `'B,C'`) that complicated encoding.
* This is a **simplified project** designed to **learn and demonstrate** the ML pipeline.
* The low performance on some classes is expected due to **class imbalance** and **small dataset size**.

---

## 💾 Saving the Model (Optional)

```python
import joblib
joblib.dump(model, 'rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(feature_columns, 'feature_columns.joblib')
```

---

## 📦 Dependencies

* pandas
* numpy
* scikit-learn
* joblib
* openpyxl (for reading Excel)

Install all using:

```bash
pip install pandas numpy scikit-learn joblib openpyxl
```

---

## 🙌 Summary

This project is a **great starting point** to understand:

* Data preprocessing (encoding, scaling)
* Model training with RandomForest
* Prediction with new inputs
* Handling categorical columns and feature mismatch

The pipeline here can be scaled and enhanced with better datasets, model tuning, and advanced validation methods.

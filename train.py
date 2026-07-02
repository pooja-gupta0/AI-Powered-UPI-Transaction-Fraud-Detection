import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

try:
    df = pd.read_csv('Upi Fraud detection.csv')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error: Dataset not found! {e}")
    exit()

# 2. Features 
features = [
    'agent_type', 'amount', 'upi_app', 'bank', 'status', 
    'amount_slab', 'hour_of_day', 'is_night_transaction', 
    'is_weekend', 'attempt_count'
]
target = 'is_suspicious'

X = df[features]
y = df[target]

# 3. Data Preprocessing 
numeric_features = ['amount', 'hour_of_day', 'attempt_count']
categorical_features = ['agent_type', 'upi_app', 'bank', 'status', 'amount_slab', 'is_night_transaction', 'is_weekend']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Pipeline  (With Balanced Class Weight)
# 'class_weight=balanced' se model fraud data ko zyada importance dega
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        random_state=42
    ))
])

# 5. Model Training
print("Training model... please wait.")
clf.fit(X, y)

# 6. Model Save
joblib.dump(clf, 'fraud_model.pkl')
print(" Done! 'fraud_model.pkl' created.")
print("Restart 'app.py' to use the updated model.")

from sklearn.metrics import classification_report, accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

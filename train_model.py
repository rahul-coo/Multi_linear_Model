import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
print("Loading dataset...")
california = fetch_california_housing()
dataset = pd.DataFrame(california.data, columns=california.feature_names)
dataset['Price'] = california.target

# Split features and target
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standard Scaling
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training model...")
regression = LinearRegression()
regression.fit(X_train_scaled, y_train)

# Save artifacts
print("Saving artifacts...")
pickle.dump(regression, open('regressor.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Done! Artifacts 'regressor.pkl' and 'scaler.pkl' created.")

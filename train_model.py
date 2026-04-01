import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset with multi-level header fix
data = pd.read_csv("stock_data.csv", header=[0,1])

# Fix column names
data.columns = data.columns.get_level_values(0)

print("Columns:", data.columns)

# Use Close column
data = data[['Close']]

# Remove missing values
data = data.dropna()

# Create prediction column
data['Prediction'] = data['Close'].shift(-30)

# Drop last rows
data = data.dropna()

# Features & labels
X = data[['Close']]
y = data['Prediction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully!")
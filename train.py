# Import libraries
import pandas as pd                         # For loading and handling tabular data
import pickle                               # For saving the trained model
from sklearn.ensemble import RandomForestRegressor  # The regression model we'll use
from sklearn.model_selection import train_test_split  # For splitting data

# --------------------------------------
# 1. Load and prepare the housing dataset
# --------------------------------------

# Load the Housing.csv file
df = pd.read_csv("Housing.csv")

# Convert yes/no categorical columns to 1/0 (binary encoding)
df["mainroad"] = df["mainroad"].map({"yes": 1, "no": 0})
df["guestroom"] = df["guestroom"].map({"yes": 1, "no": 0})
df["basement"] = df["basement"].map({"yes": 1, "no": 0})
df["hotwaterheating"] = df["hotwaterheating"].map({"yes": 1, "no": 0})
df["airconditioning"] = df["airconditioning"].map({"yes": 1, "no": 0})
df["prefarea"] = df["prefarea"].map({"yes": 1, "no": 0})

# Convert "furnishingstatus" into dummy/one-hot encoding (drop_first avoids duplicate info)
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# --------------------------------------
# 2. Split features (X) and target (y)
# --------------------------------------

# Features: all columns except "price"
X = df.drop("price", axis=1)

# Target: the "price" column (what we want to predict)
y = df["price"]

# --------------------------------------
# 3. Train the regression model
# --------------------------------------

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regression model
model = RandomForestRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)

# --------------------------------------
# 4. Save the trained model
# --------------------------------------

# Save the model as a file so it can be used in app.py
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Confirmation message
print("Model trained and saved to app/model.pkl")

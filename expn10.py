import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset from CSV
# Example CSV: columns = feature1, feature2, ..., label
data = pd.read_csv("data1.csv")  

# Split into features (X) and target (y)
X = data.drop("label", axis=1)   # all columns except label
y = data["label"]

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes classifier
nb = GaussianNB()

# Train
nb.fit(X_train, y_train)

# Predict
y_pred = nb.predict(X_test)

# Compute accuracy
acc = accuracy_score(y_test, y_pred)
print("Predictions:", y_pred)
print("Accuracy:", acc * 100, "%")

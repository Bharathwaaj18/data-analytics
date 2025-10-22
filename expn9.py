import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Activation
def sigmoid(x): return 1 / (1 + np.exp(-x))
def dsigmoid(x): return x * (1 - x)
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode target
enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights
np.random.seed(1)
W1 = np.random.randn(4, 5)   # 4 inputs → 5 hidden neurons
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 3)   # 5 hidden → 3 output classes
b2 = np.zeros((1, 3))

# Training
lr = 0.05
for epoch in range(1000):
    # Forward
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    # Loss (cross-entropy)
    loss = -np.mean(np.sum(y_train * np.log(a2 + 1e-8), axis=1))

    # Backprop
    d2 = a2 - y_train
    dW2 = a1.T @ d2 / len(X_train)
    db2 = np.mean(d2, axis=0, keepdims=True)

    d1 = (d2 @ W2.T) * dsigmoid(a1)
    dW1 = X_train.T @ d1 / len(X_train)
    db1 = np.mean(d1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test
z1 = X_test @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
a2 = softmax(z2)

preds = np.argmax(a2, axis=1)
true = np.argmax(y_test, axis=1)
acc = np.mean(preds == true)
print("\nTest Accuracy:", acc * 100, "%")

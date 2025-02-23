import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate dummy data
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Add some irrelevant features (noise)
X = np.hstack([X, np.random.normal(0, 1, (100, 5))])

# Lasso Regression from scratch
class LassoRegression:
    def __init__(self, lr=0.01, lambda_param=0.1, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = -(2/n_samples) * np.dot(X.T, (y - y_pred)) + self.lambda_param * np.sign(self.weights)
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train the model
model = LassoRegression(lr=0.01, lambda_param=0.5, n_iters=2000)
model.fit(X, y)

# Predictions and plotting trend line
y_pred = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted')
plt.show()

# Feature selection demonstration
important_features = np.where(model.weights != 0)[0]
print(f"Selected features (non-zero weights): {important_features}")
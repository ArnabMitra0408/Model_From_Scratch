import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 2)
true_coefficients = np.array([3, 5])
y = X @ true_coefficients + 4 + np.random.randn(100) * 0.5

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Linear Regression (Gradient Descent)
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train custom model
model_scratch = LinearRegressionScratch(learning_rate=0.05, n_iterations=1000)
model_scratch.fit(X_train, y_train)
y_pred_scratch = model_scratch.predict(X_test)

# Train sklearn model
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)

# Performance comparison
mse_scratch = mean_squared_error(y_test, y_pred_scratch)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

print(f"MSE (Scratch Model): {mse_scratch:.4f}")
print(f"MSE (Sklearn Model): {mse_sklearn:.4f}")

# Visualization for comparison
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_scratch, color='blue', label='Scratch Model Predictions', alpha=0.8)
plt.scatter(y_test, y_pred_sklearn, color='red', label='Sklearn Model Predictions', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Linear Regression Models')
plt.legend()
plt.show()

import joblib
from sklearn.datasets import load_iris

# Load model
model = joblib.load("knn_model.pkl")

# Example input (first 5 samples)
iris = load_iris()
X = iris.data[:5]

predictions = model.predict(X)
print("✅ Predictions for sample inputs:", predictions)

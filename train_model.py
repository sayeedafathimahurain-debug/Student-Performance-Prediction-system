import pickle
from sklearn.linear_model import LinearRegression

# Training data
X = [
    [2, 50],
    [3, 60],
    [4, 70],
    [5, 80],
    [6, 90],
    [7, 95],
    [8, 98]
]

# Scores
y = [35, 45, 55, 70, 85, 92, 98]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model created successfully!")
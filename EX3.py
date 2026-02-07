# IMPLEMENTATION OF BACKPROPAGATION ALGORITHM
# TO BUILD AN ARTIFICIAL NEURAL NETWORK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 1. GENERATE DATASET
data = {
    'Sqft': [1500, 2000, 1200, 2500, 1800],
    'Beds': [3, 4, 2, 4, 3],
    'Price': [300000, 450000, 250000, 550000, 380000]
}
df = pd.DataFrame(data)
df.to_csv('house_prices.csv', index=False)
# 2. LOAD AND NORMALIZE DATA
X = df[['Sqft', 'Beds']].values
y = df[['Price']].values
x_max = X.max(axis=0)
y_max = y.max()
X_scaled = X / x_max
y_scaled = y / y_max
# 3. ACTIVATION FUNCTIONS
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# 4. INITIALIZE WEIGHTS AND BIAS
np.random.seed(42)
# Weights
w_h = np.random.uniform(size=(2, 4))   # Input → Hidden
w_o = np.random.uniform(size=(4, 1))   # Hidden → Output
# Bias
b_h = np.random.uniform(size=(1, 4))
b_o = np.random.uniform(size=(1, 1))
learning_rate = 0.1
epochs = 5000
loss_history = []
# 5. TRAINING USING BACKPROPAGATION
print("Training the Neural Network...\n")
for epoch in range(epochs):
    # ---------- Forward Pass ----------
    hidden_input = np.dot(X_scaled, w_h) + b_h
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w_o) + b_o
    prediction_scaled = final_input   # Linear output for regression
    # ---------- Error Calculation ----------
    error = y_scaled - prediction_scaled
    loss = np.mean(error ** 2)   # Mean Squared Error
    loss_history.append(loss)
    # ---------- Backpropagation ----------
    d_output = error
    error_hidden = d_output.dot(w_o.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    # ---------- Weight Updates ----------
    w_o += hidden_output.T.dot(d_output) * learning_rate
    w_h += X_scaled.T.dot(d_hidden) * learning_rate
    # ---------- Bias Updates ----------
    b_o += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    b_h += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
# 6. FINAL PREDICTIONS
final_predictions = prediction_scaled * y_max
print("--- Final House Price Predictions ---")
for i in range(len(X)):
    print(f"Actual: ₹{int(y[i][0])} | Predicted: ₹{int(final_predictions[i][0])}")
# 7. PLOT LEARNING CURVE
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Error Reduction Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
def lowess(x, y, f):
    n = len(x)
    r = int(np.ceil(f * n))
    yest = np.zeros(n)
    for i in range(n):
        dist = np.abs(x - x[i])
        h = np.sort(dist)[r]
        w = (1 - (dist / h) ** 3) ** 3
        w[dist > h] = 0
        X = np.vstack([np.ones(n), x]).T
        W = np.diag(w)
        beta = solve(X.T @ W @ X, X.T @ W @ y)
        yest[i] = beta[0] + beta[1] * x[i]
    return yest
# Generate sample data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + 0.3*np.random.randn(100)
# LOWESS fitting
y_pred = lowess(x, y, 0.25)
# Plot
plt.scatter(x, y, color="red")
plt.plot(x, y_pred, color="blue")
plt.title("Locally Weighted Regression")
plt.show()
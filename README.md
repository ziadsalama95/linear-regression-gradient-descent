# Linear Regression with Gradient Descent

This repository contains a Jupyter Notebook that demonstrates how to implement **Linear Regression** using **Gradient Descent** from scratch. The goal is to fit a linear model to a dataset by minimizing the **Mean Squared Error (MSE)** cost function. This notebook is designed to help students understand the fundamentals of linear regression, gradient descent, and optimization.

---

## Introduction

Linear regression is one of the simplest and most widely used machine learning algorithms. It models the relationship between a dependent variable $y$ and one or more independent variables $X$ by fitting a linear equation to the observed data. In this notebook, we implement **gradient descent**, an optimization algorithm, to find the optimal parameters (slope $m$ and intercept $b$) for the linear model.

---

## Key Concepts

### Linear Regression
Linear regression assumes a linear relationship between the input variables $X$ and the output variable $y$. The model is represented as:

$$
y = mX + b
$$

Where:
- $m$ is the slope (weight) of the line.
- $b$ is the intercept (bias).

![Linear Regression](https://miro.medium.com/v2/resize:fit:1400/1*3CP1HOTBDtUtyDB50NtHXA.png)

---

### Gradient Descent
Gradient descent is an iterative optimization algorithm used to minimize a cost function. It works by updating the model parameters in the opposite direction of the gradient of the cost function with respect to the parameters.

The update rule for gradient descent is:

$$
\theta = \theta - \alpha \cdot \nabla J(\theta)
$$

Where:
- $\theta$ represents the parameters ($m$ and $b$).
- $\alpha$ is the learning rate.
- $\nabla J(\theta)$ is the gradient of the cost function.

![Gradient Descent](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/09/631731_P7z2BKhd0R-9uyn9ThDasA.webp)

---

### Mean Squared Error (MSE)
The Mean Squared Error (MSE) is a common cost function used in regression problems. It measures the average squared difference between the predicted and actual values:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $y_i$ is the actual value.
- $\hat{y}_i$ is the predicted value.
- $n$ is the number of data points.

---

## Notebook Walkthrough

### Data Generation
We generate synthetic data for linear regression using the equation:

$$
y = 4 + 3X + \text{noise}
$$

The data is visualized using a scatter plot.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualize the data
plt.scatter(X, y, color='blue', label='Data points')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Data')
plt.legend()
plt.show()
```

---

### Cost Function and Gradient Calculation
We define the `compute_cost` function to calculate the MSE and the `compute_gradient` function to compute the gradients of the cost function with respect to the parameters.

```python
def compute_cost(theta, X, y):
    predictions = X.dot(theta)
    cost = (1 / len(X)) * np.sum((predictions - y) ** 2)
    return cost

def compute_gradient(theta, X, y):
    error = X.dot(theta) - y
    gradients = (2 / len(X)) * X.T.dot(error)
    return gradients
```

---

### Gradient Descent Algorithm
We implement the gradient descent algorithm to iteratively update the parameters $m$ and $b$ until convergence.

```python
# Gradient descent parameters
learning_rate = 0.1
iterations = 1000
theta_init = np.random.randn(2, 1)  # Initial guess for m and b
tolerance = 1e-6

# Perform gradient descent
theta = theta_init
cost_history = []
iterations_done = 0

# Add a bias column to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]

for i in range(iterations):
    gradients = compute_gradient(theta, X_b, y)
    theta = theta - learning_rate * gradients
    cost = compute_cost(theta, X_b, y)
    cost_history.append(cost)

    # Stopping condition
    if i > 0 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
        print(f"Stopping early due to small change in cost.")
        break

# Optimal parameters
optimal_theta = theta
print(f"Optimal parameters (m, b): {optimal_theta.flatten()}")
print(f"Final cost: {cost_history[-1]}")
```

---

### Visualizing Convergence
We plot the cost function over iterations to visualize the convergence of gradient descent.

```python
plt.plot(cost_history, linestyle='dashed')
plt.scatter(range(iterations_done), cost_history, color='red', label='Iteration costs')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Convergence of Gradient Descent')
plt.legend()
plt.show()
```

---

### Best-Fit Line
Finally, we plot the best-fit line using the optimal parameters found by gradient descent.

```python
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X_b.dot(optimal_theta), color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Best Fit Line')
plt.legend()
plt.show()
```

---

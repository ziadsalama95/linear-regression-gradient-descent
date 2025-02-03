# [Gradient Descent for Linear Regression](https://github.com/ziadsalama95/linear-regression-gradient-descent/blob/main/Gradient_Descent_Linear_Regression.ipynb)

Welcome! In this notebook, we’ll explore **Linear Regression** and how to fit a line to data using **[Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md)**. By the end of this notebook, you’ll understand how to:
1. Define a linear model.
2. Use [Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md) to minimize the **Mean Squared Error (MSE)**.
3. Visualize the convergence of the algorithm and the best-fit line.

Let’s get started!

---

## What is Linear Regression?

Linear Regression is a statistical method used to model the relationship between a dependent variable $y$ and one or more independent variables $X$. The goal is to find the best-fit line that minimizes the difference between the predicted and actual values.

The linear model is represented as:

$$
y = mX + b
$$

Where:
- $m$ is the **slope** of the line.
- $b$ is the **intercept** (where the line crosses the y-axis).

Our task is to find the optimal values of $m$ and $b$ that minimize the **Mean Squared Error (MSE)**.

---

## Generate and Visualize the Data

We’ll start by generating some synthetic data for our linear regression problem. The data is generated using the equation:

$$
y = 4 + 3X + \text{noise}
$$

Here’s the code to generate and visualize the data:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Visualize the data
plt.scatter(X, y, color='blue', label='Data points')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Data')
plt.legend()
plt.show()
```

### Output:
![Linear Regression Data](https://raw.githubusercontent.com/ziadsalama95/linear-regression-gradient-descent/refs/heads/main/Linear%20Regression%20Data.png)

---

## Define the Cost Function (MSE)

The **Mean Squared Error (MSE)** is the average squared difference between the predicted values $\hat{y}$ and the actual values $y$. It’s defined as:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

For our linear model $\hat{y} = mX + b$, the MSE becomes:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mX_i + b))^2
$$

Here’s the Python function to compute the MSE:

```python
def compute_cost(theta, X, y):
    predictions = X.dot(theta)
    cost = (1 / len(X)) * np.sum((predictions - y) ** 2)
    return cost
```

---

## Compute the Gradient

The gradient of the cost function tells us how to adjust the parameters $m$ and $b$ to minimize the MSE. The gradients with respect to $m$ and $b$ are:

$$
\frac{\partial MSE}{\partial m} = \frac{2}{n} \sum_{i=1}^{n} X_i \cdot (mX_i + b - y_i)
$$

$$
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (mX_i + b - y_i)
$$

Here’s the Python function to compute the gradient:

```python
def compute_gradient(theta, X, y):
    error = X.dot(theta) - y
    gradients = (2 / len(X)) * X.T.dot(error)
    return gradients
```

---

## Implement [Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md)

[Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md) is an iterative algorithm that updates the parameters $m$ and $b$ to minimize the MSE. The update rule is:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla J(\theta_{\text{old}})
$$

Where:
- $\theta = [b, m]$ are the parameters.
- $\alpha$ is the **learning rate**.
- $\nabla J(\theta)$ is the gradient of the cost function.

Here’s the implementation:

```python
# Gradient descent parameters
learning_rate = 0.1
iterations = 1000
theta_init = np.random.randn(2, 1)  # Initial guess for m and b
tolerance = 1e-6

# Add a bias column (ones) to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Perform gradient descent
theta = theta_init
cost_history = []
iterations_done = 0

for i in range(iterations):
    gradients = compute_gradient(theta, X_b, y)
    theta = theta - learning_rate * gradients  # Update parameters
    cost = compute_cost(theta, X_b, y)
    cost_history.append(cost)

    # Stopping condition
    if i > 0 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
        print("Stopping early due to small change in cost.")
        break

# Optimal parameters
optimal_theta = theta
print(f"Optimal parameters (m, b): {optimal_theta.flatten()}")
print(f"Final cost: {cost_history[-1]}")
```

### Output:
```
Stopping early due to small change in cost.
Optimal parameters (m, b): [4.20851977 2.77591998]
Final cost: 0.8065976258942447
```

---

## Visualize the Convergence

Let’s plot the cost function over iterations to see how [Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md) converges:

```python
plt.plot(cost_history, linestyle='dashed')
plt.scatter(range(iterations_done), cost_history, color='red', label='Iteration costs')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Convergence of Gradient Descent')
plt.legend()
plt.show()
```

### Output:
![Convergence of Gradient Descent](https://raw.githubusercontent.com/ziadsalama95/linear-regression-gradient-descent/refs/heads/main/Convergence%20of%20Gradient%20Descent.png)

---

## Watch Gradient Descent in Action!

Here’s a cool animation showing how gradient descent works its magic:

![Gradient Descent Animation](https://raw.githubusercontent.com/ziadsalama95/linear-regression-gradient-descent/refs/heads/main/gradient_descent.gif)

What’s happening here:
- The **red line** is the current guess for the best-fit line.
- The **green dashed line** is the final, optimal fit.
- You can see the cost (MSE) decreasing as the algorithm improves the fit.

It’s like watching the line "learn" where it should be!

---

## Plot the Best-Fit Line

Finally, let’s plot the best-fit line using the optimal parameters found by [Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md):

```python
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X_b.dot(optimal_theta), color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Best Fit Line')
plt.legend()
plt.show()
```

### Output:
![Best-Fit Line](https://raw.githubusercontent.com/ziadsalama95/linear-regression-gradient-descent/main/Linear%20Regression%20Best%20Fit%20Line.png)

---

## Key Takeaways

1. **[Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md):** An iterative algorithm to minimize the cost function by adjusting the parameters in the direction of the steepest descent.
2. **Learning Rate ($\alpha$):** Controls the step size. If it’s too large, the algorithm may overshoot the minimum; if it’s too small, convergence will be slow.
3. **Mean Squared Error (MSE):** Measures the average squared difference between the predicted and actual values.
4. **Best-Fit Line:** The line that minimizes the MSE and best represents the relationship between $X$ and $y$.

---

## Experiment and Explore!

Feel free to experiment with:
- Different learning rates.
- Different initial values for $m$ and $b$.
- Different datasets.

Observe how these changes affect the convergence of [Gradient Descent](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/README.md) and the best-fit line.

---

Happy learning! 🚀

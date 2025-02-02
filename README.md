# Gradient Descent for Linear Regression

This notebook demonstrates how **Gradient Descent** can be used to find the optimal parameters (slope and intercept) for a **linear regression** model. The goal is to minimize the **Mean Squared Error (MSE)** between the predicted values and the actual values in a dataset.

The notebook covers the following concepts:
- **Data Generation**: Creating random data points to simulate a linear relationship.
- **Cost Function (MSE)**: Understanding how the MSE is used to evaluate how well the model predicts the target values.
- **Gradient Computation**: Calculating the gradient of the cost function to figure out how to adjust the model parameters.
- **Gradient Descent Algorithm**: Iteratively updating the slope and intercept to minimize the MSE.
- **Convergence Visualization**: Visualizing how the cost decreases over iterations as gradient descent optimizes the parameters.
- **Best-Fit Line**: Plotting the best-fit line after training the model.

## How It Works:

1. **Data Generation**: A synthetic dataset is created where the target variable $\hat{y}$ is related to the independent variable $X$ using a linear equation, with some added noise for realism.

2. **Cost Function**: The **Mean Squared Error (MSE)** is calculated to measure the difference between the actual and predicted values. The MSE is given by:
   
   $$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - (mx_i + b))^2$$

   
4. **Gradient Descent**: The gradient of the MSE cost function is computed with respect to the model parameters (slope $m$ and intercept $b$). By adjusting these parameters in the direction of the negative gradient, we minimize the cost.

5. **Optimization**: Gradient descent works by repeatedly updating the parameters, reducing the cost each time. The algorithm stops when the change in cost becomes negligible, indicating that the optimal parameters have been found.

6. **Visualization**: As the gradient descent algorithm runs, we plot how the cost decreases, showing the progress of the optimization. At the end, we visualize the best-fit line obtained by the optimized parameters.

## Results:

After running the gradient descent algorithm, you will see:
- The optimal values for the slope $m$ and intercept $b$ that minimize the MSE.
- A plot showing how the cost decreases with each iteration.
- A plot of the best-fit line that best represents the relationship between $X$ and $y$.

This notebook provides a hands-on introduction to using gradient descent for linear regression, helping you understand how machine learning algorithms learn from data.

import numpy             as np
import matplotlib.pyplot as plt
import globals           as gb
from os                  import path

data_matrix = np.loadtxt(path.join(gb.DATASET_DIR, 'GPUBenchmark.csv'), delimiter=',')
X, y        = data_matrix[:, :6], data_matrix[:, 6]

# Start by normalizing X using Xn = (X − μ)/σ.
X_fn = np.zeros((X.shape))
for index, column in enumerate(X.T):
    X_fn[:, index] = gb.feature_normalization(column)

# Multivariate datasets are hard to visualize. However, to get a basic understanding it might
# be a good idea to produce a plot Xi vs y for each one of the features
for index, column in enumerate(X_fn.T):
    sp = plt.subplot(2, 3, index + 1)
    sp.scatter(column, y, marker='.')
plt.show(block=False)

# Compute β using the normal equation
X_fn_e      = np.c_[np.ones((X_fn.shape[0], 1)), X_fn]
beta_X_fn_e = gb.normal_equation(X_fn_e, y)

# What is the predicted benchmark result for a graphic card
# with the following (non-normalized) feature values?
# 2432, 1607, 1683, 8, 8, 256
# The actual benchmark result is 114.
SAMPLE_GRAPHIC_CARD = np.array([[2432, 1607, 1683, 8, 8, 256]])
X_sgc               = np.insert(X, 0, SAMPLE_GRAPHIC_CARD, 0)
X_sgc_fn            = np.zeros(X_sgc.shape)
for index, column in enumerate(X_sgc.T):
    X_sgc_fn[:, index] = gb.feature_normalization(column)
X_sgc_fn_e      = np.c_[np.ones((X_sgc_fn.shape[0], 1)), X_sgc_fn]
print(f"Predicted benchmark result from normal equation for graphic card {SAMPLE_GRAPHIC_CARD}:", X_sgc_fn_e.dot(beta_X_fn_e)[0])

## What is the cost J(β) when using the β computed by the normal equation above?
cost_X_fn_e = gb.cost_function(X_fn_e, y, beta_X_fn_e)
print(f"Cost J({beta_X_fn_e}): {cost_X_fn_e}")

# Gradient descent
# (a) Find (and print) hyperparameters (α, N ) such that you get within 1% of the final cost
# for the normal equation.
# (b) What is the predicted benchmark result for the example graphic card presented above?
LEARNING_RATE = 0.027
# Runs until within 1% of actual_beta.
cost, beta_gd, iterations = gb.gradient_descent(X_fn_e, y, actual_beta=beta_X_fn_e, learning_rate=LEARNING_RATE) 
result = f"""
Cost: {cost}
Beta: {beta_gd}
Iterations: {iterations}
Learning rate: {LEARNING_RATE}"""
print(result)
## Tested some values for α/learning rate (can be seen in results.txt).
print(f"Predicted benchmark result from gradient descent for graphic card {SAMPLE_GRAPHIC_CARD}:", X_sgc_fn_e.dot(beta_gd)[0])

plt.show()
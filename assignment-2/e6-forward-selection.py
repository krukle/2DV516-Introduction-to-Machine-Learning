import numpy                 as np
import globals               as gb
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import cross_val_predict
from os                      import path

data = np.loadtxt(path.join(gb.DATASET_DIR, 'GPUBenchmark.csv'), delimiter=',')
X, y = data[:, :6], data[:, 6]

# Implement the forward selection algorithm as discussed in Lecture 6 (see lecture notes for
# details). In 
# the loop use the training MSE to find the best model in each iteration. The
# algorithm should produce p + 1 models M0, . . . , Mp, where Mi is the best model using
# i features. In terms of output, an alternative could be to let the algorithm produce a
# p-dimensional vector where its first entry is the feature in M1, its second entry is the new
# feature in M2 etc. 
# Apply your forward selection on the GPUbenchmark.csv.
best_models = gb.forward_selection(X, y)

# Use 3-fold cross-validation to find
# the best model among all Mi, i = 1, . . . , 6. 
mse = np.full((len(best_models)), np.finfo(np.float64).max)
for ix, model in enumerate(best_models):
    mse[ix] = mean_squared_error(y, cross_val_predict(LinearRegression(), model, y, cv=3))
for ix, value in enumerate(mse):
    print(f"Model number {ix}'s 3-fold cross-validation MSE: {round(value, 2)}")

# Which is the best model? Which is the most important feature, i.e. selected first?
print(f"""
The best model is model number {mse.argmin()} with {best_models[mse.argmin()].shape[1]} features.
Those features are:
{best_models[mse.argmin()]}

The most important feature is {best_models[0].T}
""")
import numpy                 as np
import matplotlib.pyplot     as plt
import globals               as gb
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import cross_val_predict
from matplotlib.colors       import ListedColormap
from os                      import path

data = np.loadtxt(path.join(gb.DATASET_DIR, 'microchips.csv'), delimiter=',')
X, y = data[:, :2], data[:, 2]

# Use Logistic regression and mapFeatures from the previous exercise to construct nine
# different classifiers, one for each of the degrees d ∈ [1, 9], and produce a figure containing a
# 3 × 3 pattern of subplots showing the corresponding decision boundaries. Make sure that
# you pass the argument C=10000.2
STEP_SIZE    = .008 # step size in the mesh
cmap_light   = ListedColormap(['#FFAAAA', '#AAAAFF']) # mesh plot
cmap_bold    = ListedColormap(['#FF0000', '#0000FF']) # colors
x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
xx, yy       = np.meshgrid(np.arange(x_min, x_max, STEP_SIZE), np.arange(y_min, y_max, STEP_SIZE)) # Mesh Grid
x1,x2        = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays
low_reg      = LogisticRegression(solver='lbfgs', C=10000., max_iter=1000)
errors_cross_val_lows = []
for degree in range(1, 10):
    XXe = gb.map_feature(x1, x2, degree, ones=False)
    Xe  = gb.map_feature(X[:, 0], X[:, 1], degree, ones=False)
    low_reg.fit(Xe, y)
    prediction   = low_reg.predict(Xe)
    grid_predict = low_reg.predict(XXe)
    mesh         = grid_predict.reshape(xx.shape)
    errors       = np.sum(prediction!=y)

    plt.subplot(3, 3, degree)
    plt.pcolormesh(xx, yy, mesh, cmap=cmap_light)
    plt.scatter(Xe[:, 0], Xe[:, 1], c=y, marker='.', cmap=cmap_bold)
    plt.title(f'Degree: {degree}, Training errors: {errors}')
    errors_cross_val_lows.append(np.sum(cross_val_predict(low_reg, Xe, y)!=y))
plt.show()

# Redo 1) but now use the regularization parameter C = 1. What is different than from the
# step in 1)?
high_reg  = LogisticRegression(solver='lbfgs', C=1, max_iter=1000)
errors_cross_val_highs = []
for degree in range(1, 10):
    XXe = gb.map_feature(x1, x2, degree, ones=False)
    Xe  = gb.map_feature(X[:, 0], X[:, 1], degree, ones=False)
    high_reg.fit(Xe, y)
    prediction   = high_reg.predict(Xe)
    grid_predict = high_reg.predict(XXe)
    mesh         = grid_predict.reshape(xx.shape)
    errors       = np.sum(prediction!=y)
    plt.subplot(3, 3, degree)
    plt.pcolormesh(xx, yy, mesh, cmap=cmap_light)
    plt.scatter(Xe[:, 0], Xe[:, 1], c=y, marker='.', cmap=cmap_bold)
    plt.title(f'Degree: {degree}, Training errors: {errors}')
    errors_cross_val_highs.append(np.sum(cross_val_predict(high_reg, Xe, y)!=y))
plt.show()

print("Regularization is stronger in second step, making for less variation in decision boundaries, i.e. lessening overfitting.")

# Finally, you should use cross-validation (in sklearn) to see which of the regularized and
# unregularized models performs best. The results could for instance be visualized in a graph
# where you plot the degree d vs. #errors, and differentiate regularized and unregularized
# by color.
plt.plot(range(1, 10), errors_cross_val_lows, c='blue', label="Low regularization")
plt.plot(range(1, 10), errors_cross_val_highs, c='red', label="High regularization")
plt.xlabel("Degree")
plt.ylabel("Errors")
plt.legend()
plt.show()
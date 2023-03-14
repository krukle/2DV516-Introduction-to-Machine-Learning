import numpy             as np
import matplotlib.pyplot as plt
import globals           as gb
from matplotlib.colors   import ListedColormap
from os                  import path

data_matrix = np.loadtxt(path.join(gb.DATASET_DIR, 'microchips.csv'), delimiter=',')
X, y        = data_matrix[:, :2], data_matrix[:, 2]

# Plot the data in X and y using different symbols or colors for the two different classes.
# Notice also that X1 and X2 are already normalized. Hence, no need for normalization in
# this exercise
plt.scatter(X[:, 0], y, c='blue', marker='.')
plt.scatter(X[:, 1], y, c="red", marker='.')
plt.show()

# Use gradient descent to find β in the case of a quadratic model.
N          = 100000
ALPHA      = 5
Xe         = gb.map_feature(X[:, 0], X[:, 1], 4)
beta, data = gb.vectorized_gradient_descent(Xe, y, iterations=N, learning_rate=ALPHA, extra_data=True)

# Print the hyper parameters α and n_iter, and produce a 1 × 2 plot with: 1) the cost function
# J(β) as a function over iterations, 2) the corresponding decision boundary (together with
# the X, y scatter plot), and 3) the number of training errors presented as a part of the
# decision boundary plot title.
plt.subplot(1, 2, 1)
plt.plot(range(N), data)
plt.xlabel("Iterations")
plt.ylabel("Cost J(β)")
plt.title(f"α = {ALPHA}, Iterations = {N}")

STEP_SIZE      = .008 # step size in the mesh
x_min, x_max   = Xe[:, 1].min()-0.1, Xe[:, 1].max()+0.1
y_min, y_max   = Xe[:, 2].min()-0.1, Xe[:, 2].max()+0.1
xx, yy         = np.meshgrid(np.arange(x_min, x_max, STEP_SIZE), np.arange(y_min, y_max, STEP_SIZE)) # Mesh Grid
x1,x2          = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays
XXe            = gb.map_feature(x1, x2, 4) # Extend matrix for degree.
prediction     = XXe.dot(beta)
proababilities = gb.sigmoid(prediction) # classify mesh ==> probabilities
classes        = proababilities.round() # round off probabilities
clz_mesh       = classes.reshape(xx.shape) # return to mesh format
cmap_light     = ListedColormap(['#FFAAAA', '#AAAAFF']) # mesh plot
cmap_bold      = ListedColormap(['#FF0000', '#0000FF']) # colors

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(Xe[:, 1], Xe[:, 2], c=y, marker='.', cmap=cmap_bold)
plt.title(f"Training errors: {np.sum(gb.sigmoid(Xe.dot(beta)).round() != y)}")
plt.show()

# Use mapFeatures to repeat 2) but with a polynomial of degree five (d = 5) mode
N          = 200000
Xe         = gb.map_feature(X[:, 0], X[:, 1], 5)
beta, data = gb.vectorized_gradient_descent(Xe, y, iterations=N, learning_rate=ALPHA, extra_data=True)

plt.subplot(1, 2, 1)
plt.plot(range(N), data)
plt.xlabel("Iterations")
plt.ylabel("Cost J(β)")
plt.title(f"α = {ALPHA}, Iterations = {N}")

XXe            = gb.map_feature(x1, x2, 5) # Extend matrix for degree 5.
prediction     = XXe.dot(beta)
proababilities = gb.sigmoid(prediction) # classify mesh ==> probabilities
classes        = proababilities.round() # round off probabilities
clz_mesh       = classes.reshape(xx.shape) # return to mesh format

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(Xe[:, 1], Xe[:, 2], c=y, marker='.', cmap=cmap_bold)
plt.title(f"Training errors: {np.sum(gb.sigmoid(Xe.dot(beta)).round() != y)}")
plt.show()
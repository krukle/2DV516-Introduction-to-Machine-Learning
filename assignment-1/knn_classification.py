import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from os import path
from globals import DATASET_DIR

K_VALUES           = {1, 3, 5, 7}
UNKNOWN_MICROCHIPS = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
LABELS             = {0.0: 'Fail', 1.0: 'OK'}
GRID_SIZE          = 100

def generate_grid(x_axis, y_axis):
    """Generates grid.

    Args:
        x_axis (ndarray): 1d-matrix representing x axis.
        y_axis (ndarray): 1d-matrix representing y axis.

    Returns:
        ndarray: Generated grid.
    """    
    xx, yy = np.meshgrid(x_axis, y_axis)
    cells  = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return knn_clf.predict(cells).reshape(GRID_SIZE, GRID_SIZE)

data_matrix        = np.loadtxt(path.join(DATASET_DIR, 'microchips.csv'), delimiter=',')
X, y               = data_matrix[:, :2], data_matrix[:, 2]
min_x, max_x       = min(X[:, 0]), max(X[:, 0])
min_y, max_y       = min(X[:, 1]), max(X[:, 1])
x_axis             = np.linspace(min_x, max_x, GRID_SIZE)
y_axis             = np.linspace(min_y, max_y, GRID_SIZE)
knn_clf            = KNeighborsClassifier()

plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.title("Original data")
plt.show(block=False)

fig = plt.figure()

for i, k in enumerate(K_VALUES):
    knn_clf.set_params(n_neighbors=k)
    knn_clf.fit(X, y)
    predict = knn_clf.predict(UNKNOWN_MICROCHIPS)
    grid = generate_grid(x_axis, y_axis)
    sp   = fig.add_subplot(221+i)
    sp.imshow(grid, origin='lower', extent=(min_x, max_x, min_y, max_y))
    sp.scatter(X[:,0], X[:,1], c=y, edgecolors='r')
    sp.set_title(f"k = {k}")
    print(f"""
        k = {k}
            chip1: {UNKNOWN_MICROCHIPS[0]} ==> {LABELS[predict[0]]}
            chip2: {UNKNOWN_MICROCHIPS[1]} ==> {LABELS[predict[1]]}
            chip3: {UNKNOWN_MICROCHIPS[2]} ==> {LABELS[predict[2]]}
    """)

plt.show()
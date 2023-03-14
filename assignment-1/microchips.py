import numpy as np
import matplotlib.pyplot as plt
from os import path
from globals import DATASET_DIR

K_VALUES    = [1, 3, 5, 7]
GRID_SIZE   = 100
LABELS      = {0.0: 'Fail', 1.0: 'OK'}
COLORS      = {0.0: 'red', 1.0: 'blue'}

def generate_grid(x_axis, y_axis, X, labels):
    """Generates grid.

    Args:
        x_axis (ndarray): x-axis for grid.
        y_axis (ndarray): y-axis for grid.
        X (ndarray): Feautures.
        labels (ndarray): labels for features X.

    Returns:
        ndarray: 4 grids, for each value of k (1, 3, 5, 7)
    """    
    grid_1 = np.zeros(shape=(len(x_axis), len(y_axis)))
    grid_3 = np.zeros(shape=(len(x_axis), len(y_axis)))
    grid_5 = np.zeros(shape=(len(x_axis), len(y_axis)))
    grid_7 = np.zeros(shape=(len(x_axis), len(y_axis)))
    for ix, vx in enumerate(x_axis):
        for iy, vy in enumerate(y_axis):
            grid_1[ix, iy] = predict_result(calculate_distances(np.array([vx, vy]), X, labels), 1)
            grid_3[ix, iy] = predict_result(calculate_distances(np.array([vx, vy]), X, labels), 3)
            grid_5[ix, iy] = predict_result(calculate_distances(np.array([vx, vy]), X, labels), 5)
            grid_7[ix, iy] = predict_result(calculate_distances(np.array([vx, vy]), X, labels), 7)
    return grid_1.T, grid_3.T, grid_5.T, grid_7.T

def predict_result(distances, k):
    """Predicts result by returning the dominant label of the k closest elements.

    Args:
        distances (ndarray): Elements sorted by ascending order with respect to distance from a ceratin element.
        k (int): amount of elements to check.

    Returns:
        float: most dominant label in k closest elements.
    """    
    return max(set(distances[:k, 3]), key=distances[:k, 3].tolist().count)

def eucledian_distance(x, y):
    """Calculates eucledian distance.

    Args:
        x (ndarray): Vector.
        y (ndarray): Vector.

    Returns:
        float: Distance between vectors.
    """    
    return np.linalg.norm(x-y)

def manhattan_distance(x, y):
    """Calculates manhattan distance.

    Args:
        x (ndarray): Vector.
        y (ndarray): Vector.

    Returns:
        float: Distance between vectors.
    """    
    distance = 0
    for z, q in zip(x, y):
        distance += np.abs(z - q)
    return distance

def calculate_distances(source, destinations, labels):
    """Calculates distance between source and all destinations.

    Args:
        source (ndarray): Source vector.
        destinations (ndarray): Destination vectors/matrix.
        labels (ndarray): Labels for vectors.

    Returns:
        ndarray: Matrix with columns x_destination, y_destination, distance_source_to_destincation and label for destination.
    """    
    distances = np.zeros((0, 4))
    for destination, label in zip(destinations, labels):
        distances = np.append(distances, [[destination[0], destination[1], eucledian_distance(destination, source), label]], axis=0)
    return distances[distances[:, 2].argsort()]

def calculate_training_errors(X, y):
    """Calculates error per k.

    Args:
        X (ndarray): Features.
        y (ndarray): Labels.

    Returns:
        dict: Key: k-value, Value: n_training_errors
    """    
    training_errors = {K_VALUES[0] : 0, K_VALUES[1] : 0, K_VALUES[2] : 0, K_VALUES[3] : 0}
    for k in K_VALUES:        
        for u, z in zip(X, y):
            if predict_result(calculate_distances(u, X, y), k) != z:
                training_errors[k] += 1
    return training_errors

data_matrix = np.loadtxt(path.join(DATASET_DIR, 'microchips.csv'), delimiter=',')
X, y        = data_matrix[:, :2], data_matrix[:, 2]

## 1. 
for i in np.unique(y):
    ix = np.where(y == i)
    plt.scatter(X[:, 0][ix], X[:, 1][ix], c=COLORS[i], label=LABELS[i])
plt.legend()
plt.show(block=False)

## 2.
UNKNOWN_MICROCHIPS = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])

for unknown_microchip in UNKNOWN_MICROCHIPS:
    print()
    print("Microchip:" ,unknown_microchip)
    distances = calculate_distances(unknown_microchip, X, y)
    for k in K_VALUES: 
        print("k =", k, "==>", end=" ")
        print(LABELS[predict_result(distances, k)])

## 3.
min_x, max_x    = min(X[:, 0]), max(X[:, 0])
min_y, max_y    = min(X[:, 1]), max(X[:, 1])
x_axis          = np.linspace(min_x, max_x, GRID_SIZE)
y_axis          = np.linspace(min_y, max_y, GRID_SIZE)
grids           = generate_grid(x_axis, y_axis, X, y)
training_errors = calculate_training_errors(X, y)   
fig             = plt.figure()

for i in range(4):
    ax = fig.add_subplot(i+221)
    ax.imshow(grids[i], origin='lower', extent=(min_x, max_x, min_y, max_y))
    ax.scatter(X[:,0], X[:,1], c=y, edgecolors='r')
    ax.set_title(f'K = {K_VALUES[i]}, Training errors = {training_errors[K_VALUES[i]]}')
plt.show(block=True)
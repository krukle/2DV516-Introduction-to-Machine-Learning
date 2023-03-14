import numpy             as np
import matplotlib.pyplot as plt
from os                  import path
from globals             import DATASET_DIR

K_VALUES    = {1, 3, 5, 7, 9, 11}
GRID_SIZE   = 200

def mean_squared_error(A, B, ax=0):
    """Calculate Mean Square Error.
    
    Source: https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy

    Args:
        A (ndarray): Actual labels (y).
        B (ndarray): Calculated labels (f(X)).
        ax (int): Axis to calculate mean on.

    Returns:
        float: The Mean Square Error value.
    """    
    return (np.square(A-B)).mean(axis=ax)

def find_nearest(array, values, k):
    """Finds the nearest neighbors to values from array.

    Args:
        array (ndarray): Array of neighbors.
        values (ndarray): Values to find neighbors for.
        k (int): Amount of neighbors to find.

    Returns:
        ndarray: x-value, calculated y-value.
    """    
    result = np.zeros((0, 2))
    array  = np.asarray(array)
    for value in values:
        array_minus_neighbours = array
        neighbors              = np.zeros((0, 2))
        ix                     = (np.abs(array[:, 0] - value)).argmin()
        for c in range(k):   
            index                  = (np.abs(array_minus_neighbours[:, 0] - value)).argmin()
            neighbors              = np.append(neighbors, [array_minus_neighbours[index]], axis=0)
            array_minus_neighbours = np.delete(array_minus_neighbours, index, axis=0)
        result = np.append(result, [[array[ix][0], neighbors.mean(axis=0)[1]]], axis=0)
    return result

## 1.
data_matrix = np.loadtxt(path.join(DATASET_DIR, 'polynomial200.csv'), delimiter=',')
train_set   = data_matrix[100:, :]
test_set    = data_matrix[:100, :]
train_set   = train_set[train_set[:, 0].argsort()]
test_set    = test_set[test_set[:, 0].argsort()]

## 2.
fig = plt.figure()
sp1 = fig.add_subplot(121)
sp2 = fig.add_subplot(122)

sp1.scatter(train_set[:, 0], train_set[:, 1])
sp2.scatter(test_set[:, 0], test_set[:, 1])
sp1.set_title('Training set')
sp2.set_title('Testing set')

plt.show(block=False)

## 3.
min_x, max_x = min(train_set[:, 0]), max(train_set[:, 0])
x_axis       = np.linspace(min_x, max_x, GRID_SIZE)
fig          = plt.figure()

for i, k in enumerate(K_VALUES):
    train_result     = find_nearest(train_set, x_axis, k)
    train_mse_result = find_nearest(train_set, train_set[:, 0], k)
    train_mse        = mean_squared_error(train_set[:, 1], train_mse_result[:, 1])
    test_mse         = mean_squared_error(test_set[:, 1], train_mse_result[:, 1])
    sp               = fig.add_subplot(231 + i)

    sp.scatter(train_set[:, 0], train_set[:, 1], marker='.')
    sp.plot(train_result[:, 0], train_result[:, 1])
    sp.set_title(f'k = {k}, MSE = {np.round(train_mse, decimals=2)}')

    print(f"MSE on test_set for k = {k}: {np.round(test_mse, decimals=2)}")

plt.show()
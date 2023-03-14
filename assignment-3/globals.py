import numpy as np
from sklearn.model_selection import train_test_split

DATASETS_DIRECTORY = 'datasets'

def generate_X_y(data):
    """Generates X and y from dataset.

    Args:
        data (NDArray): Dataset.

    Returns:
        NDArray: X as MxN array.
        NDArray: Labels y as Mx1 array.
    """    
    return data[:, :data.shape[1]-1], data[:, data.shape[1]-1]

def generate_sets(X, y, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    """Generates training, test and optionally validation set from data X and labels y.
    Ratios should in total sum 1.0.

    Source: https://datascience.stackexchange.com/a/53161

    Args:
        X (NDArray): Data.
        y (NDArray): Labels.
        train_ratio (float, optional): Ratio for train set. Defaults to 0.6.
        validation_ratio (float, optional): Ratio for validation set. Defaults to 0.2.
        test_ratio (float, optional): Ratio for test set. Defaults to 0.2.
        
    Returns:
        NDArray : X_train
        NDArray : y_train
        NDArray : X_test
        NDArray : y_test
        NDArray, optional: X_val
        NDArray, optional: y_val
    """    
    assert 0.0 < train_ratio < 1.0, f"train_ratio has to be in range (0.0, 1.0) it is {train_ratio}."
    assert 0.0 < test_ratio < 1.0, f"test_ratio has to be in range (0.0, 1.0) it is {test_ratio}."
    ratio_sum = train_ratio + test_ratio + validation_ratio
    assert ratio_sum == 1.0, f"Sum of ratios has to equal 1.0, they equal {ratio_sum}."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio, random_state=1)
    if validation_ratio <= 0.0:
        return X_train, y_train, X_test, y_test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio+validation_ratio))
    return X_train, y_train, X_test, y_test, X_val, y_val

def feature_normalization(X):
    """Applies feature normalization on 1d matrix X. 
    Making data centered around 0 with a standard deviation of 1.

    Args:
        X (ndarray): 1d-Matrix.

    Returns:
        ndarray: Normalized 1d-matrix.
    """    
    return (X-X.mean())/(X.std()+np.nextafter(0, 1))

def sigmoid(M):
    """Applies the sigmoid function on each element of matrix M.

    Args:
        M (ndarray): Matrix with elements.

    Returns:
        ndarray: Matrix with sigmoided elements.
    """    

    return 1/(1 + np.exp(-M))

def logistic_cost_function(X_e, y, b):
    """Vectorized version of the logistic cost function.

    Args:
        X_e (ndarray): Extended 2D matrix of features.
        y (ndarray): 1D matrix of labels.
        b (ndarray): 1D matrix of betas.

    Returns:
        float: The resulting cost.
    """    
    g_Xb = sigmoid(X_e.dot(b))
    return -1/y.shape[0] * np.sum(np.dot(np.log(g_Xb), y.T) + np.dot(np.log(1-g_Xb), (1-y.T)))

def vectorized_gradient_descent(X_e, y, iterations = 20000000, learning_rate=0.0002, extra_data=False):
    """Calculates beta for X through vectorized gradient descent. Beta starts at 0. 

    Args:
        X_e (ndarray): 2D matrix of extended features.
        y (ndarray): 1D matrix of labels.
        iterations (int, optional): Amount of iterations. Defaults to 20000000.
        learning_rate (float, optional): 'Jump distance'. Defaults to 0.0002.
        extra_data (boolean, optional): Decides wheter the function return extra data or not.

    Returns:
        ndarray: Calculated Beta. (Value is approximate).
    """    
    beta = np.zeros((X_e.shape[1]))
    if extra_data is False:
        for i in range(iterations):
            beta = beta - (learning_rate/X_e.shape[0] * X_e.T.dot(sigmoid(X_e.dot(beta)) - y))
            # print(f"Iteration: {i}, J{beta}: {logistic_cost_function(X_e, y, beta)}")
            # print(f"Iteration: {i}, J: {logistic_cost_function(X_e, y, beta)}")
        return beta
    else:  
        data = np.zeros((0, 1))
        for i in range(iterations):
            cost = logistic_cost_function(X_e, y, beta)
            beta = beta - (learning_rate/X_e.shape[0] * X_e.T.dot(sigmoid(X_e.dot(beta)) - y))
            data = np.append(data, [cost])
            # print(f"Iteration: {i}, J{beta}: {cost}")
        return beta, data
        
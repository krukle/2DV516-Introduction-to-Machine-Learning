import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

DATASET_DIR = 'A2_datasets_2022'

def normal_equation(Xe, y):
    """Calculates beta for extended X; Xe and y.

    Args:
        y (ndarray): 1d-array.
        Xe (ndarray): array of features. (1, x1, x2)

    Returns:
        ndarray: 1d-array of beta values per column in Xe.
    """    
    return np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

def feature_normalization(X):
    """Applies feature normalization on 1d matrix X. 
    Making data centered around 0 with a standard deviation of 1.

    Args:
        X (ndarray): 1d-Matrix.

    Returns:
        ndarray: Normalized 1d-matrix.
    """    
    # return (X-X.mean())/np.linalg.norm(X)
    return (X-X.mean())/X.std()

def cost_function(Xe, y, beta):
    """Calculates cost.

    Args:
        Xe (ndarray): Extended X.
        y (ndarray): Labels.
        beta (ndarray): Beta.

    Returns:
        float: Cost.
    """    
    j = np.dot(Xe, beta) - y
    J = (j.T.dot(j))/Xe.shape[0]
    return J

def gradient_descent(X, y, actual_beta=None, iterations = 20000000, learning_rate=0.0002):
    """Calculates beta for X through gradient descent. Starts at 0. 

    Args:
        X (ndarray): Features.
        y (ndarray): Labels.
        actual_beta (ndarray, optional): Beta to aim for. Defaults to None.
        iterations (int, optional): Amount of iterations. Defaults to 20000000.
        learning_rate (float, optional): 'Jump distance'. Defaults to 0.0002.

    Returns:
        ndarray: Calculated Beta. (Value is approximate).
    """    
    
    beta = np.zeros((X.shape[1]))
    if actual_beta is not None:
        cost, counter = 0, 0
        actual_cost   = cost_function(X, y, actual_beta)
        while(cost <= (actual_cost * 0.09) or cost >= (actual_cost * 1.01)):
            beta = beta - (learning_rate * X.T.dot(X.dot(beta) - y))
            cost = cost_function(X, y, beta)
            counter += 1
            # print(f"Iteration: {counter}, Cost: {cost}, Beta: {beta}")
        return cost, beta, counter
    else:
        for i in range(iterations):
            beta = beta - (learning_rate * X.T.dot(X.dot(beta) - y))
            # print(f"Iteration: {i}, J{beta}: {cost_function(X, y, beta)}")
    return beta

def map_feature(X1, X2, d, ones=True): 
    """Maps X1, X2 to [1, X1, X2, X1**2, X1, X2, X2**2, ...] for any degree d.

    Args:
        X1 (ndarray): 1D Matrix of features.
        X2 (ndarray): 1D Matrix of features.
        degree (int): Degree of which to map X1 and X2.
        ones (boolean, optional): Decides wheter the Extended X should start with ones or not.

    Returns:
        ndarray: Extended X of chosen degree.
    """    
    if ones is True:
        one = np.ones([len(X1), 1])
        Xe = np.c_[one, X1, X2] # Start with [1,X1,X2]
    else:
        Xe = np.c_[X1, X2]
    for i in range(2, d+1):
        for j in range(0, i+1):
            Xnew = X1**(i-j)*X2**j # type (N)
            Xnew = Xnew.reshape(-1, 1) # type (N,1) required by append
            Xe = np.append(Xe, Xnew, 1) # axis = 1 ==> append column
    return Xe

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
        return beta
    else:  
        data = np.zeros((0, 1))
        for i in range(iterations):
            cost = logistic_cost_function(X_e, y, beta)
            beta = beta - (learning_rate/X_e.shape[0] * X_e.T.dot(sigmoid(X_e.dot(beta)) - y))
            data = np.append(data, [cost])
            # print(f"Iteration: {i}, J{beta}: {cost}")
        return beta, data
        
def forward_selection(X, y):
    """Forward selection. Creates n (X.shape[0]) amount of models. Where the 0th model has 1 feature, and the nth-1 model has n features.

    Args:
        X (ndarray): Features.
        y (ndarray): Labels.

    Returns:
        list: List of models 0 to n-1
    """    
    linreg      = LinearRegression()
    best_models = [np.zeros(X.shape[0])] * (X.shape[1])
    min_mse     = np.full((len(best_models)), np.finfo(np.float64).max)
    for p in range(X.shape[1]-1):
        for k in X.T:
            model = None
            if p == 0:
                model = k.reshape(-1, 1)
            elif not (any(np.equal(k,best_models[p-1].T).all(1))): 
                model = np.c_[best_models[p-1], k]
            if model is not None:
                linreg.fit(model, y)
                mse = mean_squared_error(y, linreg.predict(model))
                if min_mse[p] > mse:
                    min_mse[p] = mse
                    best_models[p] = model 
    best_models[-1] = X
    return best_models

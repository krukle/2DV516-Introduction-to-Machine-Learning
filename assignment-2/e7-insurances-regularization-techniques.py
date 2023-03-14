from matplotlib import pyplot as plt
from os         import path
import numpy    as np
import globals  as gb

def convert_sex(x):
    return x == 'male'
def convert_smoker(x):
    return x == 'yes'
def convert_region(x):
    return {'northwest': 0.0, 'northeast': 1.0, 'southwest': 2.0, 'southeast': 3.0}.get(x)

data = np.asarray(np.genfromtxt(path.join(gb.DATASET_DIR, 'insurance.csv'), 
                  delimiter=';', names=True, encoding=None,
                  converters={1: convert_sex, 4: convert_smoker, 5: convert_region}).tolist())
X, y = data[:, :7], data[:, 7]

for i, column in enumerate(X.T):
    plt.subplot(2, 4, i+1)
    plt.scatter(column, y, marker='.')
plt.show()

# Not done.
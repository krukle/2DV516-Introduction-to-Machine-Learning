from matplotlib.colors       import ListedColormap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import classification_report
from os                      import path
from sklearn.svm             import SVC
from matplotlib              import pyplot as plt
import numpy                 as np
import globals               as gb

data = np.genfromtxt(path.join(gb.DATASETS_DIRECTORY, 'mnistsub.csv'), delimiter=',')
X, y = gb.generate_X_y(data)
X_train, y_train, X_test, y_test, X_val, y_val = gb.generate_sets(X, y, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2) 

# Tune the necessary hyperparameters by for instance grid search. In this exercise we are concerned with
# the hyperparameters given in Table 1. Every hyperparameter should be tested for at least 3 values
# but you are free to add more testings.
# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

cmap_light = ListedColormap(['plum', 'lightcoral', 'paleturquoise', 'beige'])
cmap_bold = ListedColormap(['mediumpurple', 'indianred', 'steelblue', 'khaki'])

plt.subplot(2, 2, 1)
ax = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='.', cmap=cmap_bold)
plt.legend(*ax.legend_elements(), title="Classes")
plt.title("Data")

parameters = [
    {"kernel": ["linear"], "C": [5, 6, 100]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [0.001, 0.010, .025, .050]},
    {"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [3, 5, 7], "gamma": [.010, .050, .100]}
]

PADDING   = 1
STEP_SIZE = 0.04
min_X0, max_X0 = X[:, 0].min() - PADDING, X[:, 0].max() + PADDING
min_X1, max_X1 = X[:, 1].min() - PADDING, X[:, 1].max() + PADDING
grid_X0 = np.arange(min_X0, max_X0, STEP_SIZE)
grid_X1 = np.arange(min_X1, max_X1, STEP_SIZE)
xx, yy  = np.meshgrid(grid_X0, grid_X1)
grid    = np.hstack((xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)))
for i, parameter in enumerate(parameters):
    clf = GridSearchCV(SVC(), parameter)
    clf.fit(X_val, y_val)
    print(f"Best parameters set:\n{clf.best_params_}.\n")

    clf = clf.best_estimator_
    clf.fit(X_train, y_train)
    print("\nDetailed classification report:\n")
    print("The hyperparemeters are found on the validation set.")
    print("The model is trained on the train set.")
    print("The scores are computed on the test set.\n")
    print("\n", classification_report(y_test, clf.predict(X_test)))

    predict = clf.predict(grid)

    plt.subplot(2, 2, i+2)
    plt.pcolormesh(grid_X0, grid_X1, predict.reshape(xx.shape), cmap=cmap_light)
    ax = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='.', cmap=cmap_bold)
    plt.title(parameter["kernel"][0])

plt.show()
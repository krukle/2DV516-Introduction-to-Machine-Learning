#Source: https://towardsdatascience.com/growing-a-random-forest-using-sklearns-decisiontreeclassifier-24e048e8bd84
import pickle
import sys
import matplotlib
import numpy                 as np
import globals               as gb
from os                      import path
from sklearn                 import tree
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib              import pyplot           as plt

# If you want to run the calculations for yourself, 
# edit PLOT_FILE to anything but 'batplot'.
# Answers from prints are attached in report.
PLOT_FILE = "batplot"

if path.exists("pickles/" + PLOT_FILE + ".pickle"):
    ax = pickle.load(open("pickles/" + PLOT_FILE + ".pickle", 'rb'))
    plt.show()
    sys.exit("""
        Program ran using pickle model. 
        Edit final variable 'PLOT_FILE' to run program 
        without pickle and with prints to answers. 
        Answers exist also in report.md.
        """)

matplotlib.rc('font', size=6)
data = np.genfromtxt(path.join(gb.DATASETS_DIRECTORY, 'bm.csv'), delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2], test_size=0.5, random_state=0)
rng = np.random.default_rng(7)
n   = len(X_train)
r   = np.empty([n, 100], dtype=int)
XX  = np.empty([n, 2, 100])
yy  = np.empty([n, 1, 100])
yy_train  = y_train.reshape((5000, 1))

PADDING   = 0.1
STEP_SIZE = 0.1
min_X0, max_X0 = X_train[:, 0].min() - PADDING, X_train[:, 0].max() + PADDING
min_X1, max_X1 = X_train[:, 1].min() - PADDING, X_train[:, 1].max() + PADDING
grid_X0 = np.arange(min_X0, max_X0, STEP_SIZE)
grid_X1 = np.arange(min_X1, max_X1, STEP_SIZE)
grid_x, grid_y  = np.meshgrid(grid_X0, grid_X1)
grid    = np.hstack((grid_x.flatten().reshape(-1, 1), grid_y.flatten().reshape(-1, 1)))

scores = []
trees  = []
for i in range(99):
    clf = tree.DecisionTreeClassifier(random_state=7)
    r[:, i] = rng.choice(n, n, True)
    XX[:, :, i] = X_train[r[:, i], :]
    yy[:, :, i] = yy_train[r[:, i]]
    clf.fit(XX[:, :, i], yy[:, :, i])
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))
    grid_predict = clf.predict(grid).reshape(grid_x.shape)
    trees.append(clf)
    plt.subplot(10, 10, i+1)
    plt.title(f"Accuracy: {scores[i]*100:.2f}%")
    plt.axis("off")
    plt.contour(grid_X0, grid_X1, grid_predict, colors='black')

print("Max accuracy: {:.2f}%.".format(max(scores) * 100))
print("Min accuracy: {:.2f}%.".format(min(scores) * 100))
print("Average accuracy: {:.2f}%.".format(np.mean(scores) * 100))
print("Std of accuracy: {:.2f}.".format(np.std(scores) * 100))

grid_predictions = np.empty(len(grid), dtype=int)
predictions = np.empty(len(trees), dtype=int)
for j, coordinate in enumerate(grid):
    for i, t in enumerate(trees):
        predictions[i] = t.predict(coordinate.reshape(1, -1))
    grid_predictions[j] = np.bincount(predictions).argmax()

X_test_predictions = np.empty(len(X_test), dtype=int)
for j, xt in enumerate(X_test):
    for i, t in enumerate(trees):
        predictions[i] = t.predict(xt.reshape(1, -1))
    X_test_predictions[j] = np.bincount(predictions).argmax()    

plt.subplot(10, 10, 100)
plt.title(f"Accuracy: {accuracy_score(y_test, X_test_predictions)*100:.2f}%")
plt.contour(grid_X0, grid_X1, grid_predictions.reshape(grid_x.shape))
figure = plt.gcf()
figure.set_size_inches(19.20, 10.80)
plt.axis("off")
plt.savefig("img/" + PLOT_FILE + '.pdf')
pickle.dump(figure, open("pickles/" + PLOT_FILE + ".pickle", "wb"))
plt.show()
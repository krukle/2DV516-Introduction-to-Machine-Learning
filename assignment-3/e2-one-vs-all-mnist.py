#Source: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
from matplotlib              import pyplot as plt
from sklearn.metrics         import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm             import SVC
from os                      import path
import numpy                 as np
import globals               as gb
import gzip
import sys

# Source: https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
def fetch(file):
    filepath = path.join(gb.DATASETS_DIRECTORY, file)
    if path.isfile(filepath):
        with open(filepath, "rb") as f:
            data = f.read()
    else:
        sys.exit(f"File: '{file}' doesn't exist.")
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X      = fetch("train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
y      = fetch("train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
y_test = fetch("t10k-labels-idx1-ubyte.gz")[8:]

np.random.seed(7)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]

X_validation = X[59000:, :] /255.0
y_validation = y[59000:]
X_train = X[:20000, :] /255.0
y_train = y[:20000]
X_test = X_test /255.0

N_PARAM    = 6
parameters = [{
    "kernel": ["rbf"], 
    "gamma": np.logspace(-3, -1, num=N_PARAM), 
    "C": np.logspace(0, 1, num=N_PARAM)
    },]

print("Grid search running to find the best hyperparemeters...\n")
clf = GridSearchCV(SVC(), parameters)
clf.fit(X_validation, y_validation)

print("Best hyperparameters:\n")
print(clf.best_params_, "\n")
print("Grid scores on train set:\n")

means  = clf.cv_results_["mean_test_score"]
stds   = clf.cv_results_["std_test_score"]
params = clf.cv_results_["params"]
for mean, std, param in zip(means, stds, params):
    print(f"{mean:.3f} (+/-{std*2:.3f}) for {param}.")

print("\nDetailed classification report:\n")
print("The hyperparemeters are found on the validation set.")
print("The model is trained on the train set.")
print("The scores are computed on the test set.\n")
clf = clf.best_estimator_
clf.fit(X_train, y_train)
predict_1v1 = clf.predict(X_test)
accuracy_1v1 = np.sum(predict_1v1==y_test)*100.0/len(y_test)
print(classification_report(y_test, predict_1v1))
cm_1v1 = ConfusionMatrixDisplay.from_predictions(y_test, predict_1v1)
cm_1v1.ax_.set_title("One vs one.")
cm_1v1.ax_.text(
    0.05, 0.05, f"Accuracy: {round(accuracy_1v1, 2)}%", 
    transform=cm_1v1.ax_.transAxes,fontsize=14, 
    verticalalignment='bottom', 
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65))

# Second part.
predictions   = []
probabilities = []
best_pred     = np.full(y_test.shape, np.finfo(dtype=float).min)
X_test_n      = np.c_[np.ones((X_test.shape[0], 1)), X_test]
X_train_n     = np.c_[np.ones((X_train.shape[0], 1)), X_train]

for label in range(10):    
    y_train_   = y_train == label
    beta       = gb.vectorized_gradient_descent(X_train_n, y_train_, iterations=500, learning_rate=.3)
    prediction = X_test_n.dot(beta)
    best_pred  = np.fmax(best_pred, prediction)
    predictions.append(prediction)

predicted_labels = np.empty(y_test.shape)
for i in range(len(predicted_labels)):
    for label, array in enumerate(predictions):
        if array[i] == best_pred[i]:
            predicted_labels[i] = label
accuracy_1va = np.sum(predicted_labels==y_test, dtype=float)*100.0/len(y_test)
cm_1va       = ConfusionMatrixDisplay.from_predictions(y_test, predicted_labels)
cm_1va.ax_.set_title("One vs all.")
cm_1va.ax_.text(
    0.05, 0.05, f"Accuracy: {round(accuracy_1va, 2)}%", 
    transform=cm_1va.ax_.transAxes,fontsize=14, 
    verticalalignment='bottom', 
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65))
plt.show()
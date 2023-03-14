import numpy             as np
import matplotlib.pyplot as plt
import globals           as gb
from os                  import path

# Read data and shuffle the rows in the raw data matrix
data = np.loadtxt(path.join(gb.DATASET_DIR, 'breast_cancer.csv'), delimiter=',')
np.random.shuffle(data)
X, y = data[:, :9], data[:, 9]

# Replace the responses 2 and 4 with 0 and 1 and divide the dataset into a training set and
# a test set. How many observations did you allocated for testing, and why this number?
y = np.subtract(np.divide(y, 2), 1)
print("""
Train/test is shared 80/20. 
Source: Lecture 1: ML Introduction + kNN 
2DV516 Introduction to Machine Learning 
Dr Jonas Lundberg, office B3024""")
DIFF = 0.8
y_train, y_test = y[:int(y.shape[0]*DIFF)], y[int(y.shape[0]*DIFF):]
X_train, X_test = X[:int(X.shape[0]*DIFF), :], X[int(X.shape[0]*DIFF):, :]

# Normalize the training data and train a linear logistic regression model using gradient
# descent. Print the hyperparameters α and n_iterations and plot the cost function J(β) as a
# function over iterations.
X_n = np.c_[gb.feature_normalization(X[:, 0]), gb.feature_normalization(X[:, 1])]
Xn_train = np.ones((X_train.shape[0], 1))
for column in X_train.T:
    Xn_train = np.c_[Xn_train, gb.feature_normalization(column)]
N     = 20000
ALPHA = 0.5
beta_Xn_train, data_Xn_train = gb.vectorized_gradient_descent(Xn_train, y_train, iterations=N, learning_rate=ALPHA, extra_data=True)
plt.plot(range(N), data_Xn_train)
plt.title(f"α = {ALPHA}")
plt.xlabel("Iteration N")
plt.ylabel("Cost J(β)")
plt.show(block=False)

# What is the training error (number of non-correct classifications in the training data) and
# the training accuracy (percentage of correct classifications) for your model?
z = Xn_train.dot(beta_Xn_train)
probabilities = gb.sigmoid(z)
prediction = probabilities.round()
n_training_errors = np.sum(y_train!=prediction)
print(f"""
Training errors: {n_training_errors}
Training accuracy: {round((np.size(Xn_train, 0)-n_training_errors)/np.size(Xn_train, 0)*100, 2)}%""")

# What is the number of test error and the test accuracy for your model?
X_n = np.c_[gb.feature_normalization(X[:, 0]), gb.feature_normalization(X[:, 1])]
Xn_test = np.ones((X_test.shape[0], 1))
for column in X_test.T:
    Xn_test = np.c_[Xn_test, gb.feature_normalization(column)]
q = Xn_test.dot(beta_Xn_train)
probabilities = gb.sigmoid(q)
prediction = probabilities.round()
n_test_errors = np.sum(y_test!=prediction)
print(f"""
Test errors: {np.sum(y_test!=prediction)}
Test accuracy: {round((np.size(Xn_test, 0)-n_test_errors)/np.size(Xn_test, 0)*100, 2)}%""")

plt.show(block=True)

# Question 6 answered in report.md.
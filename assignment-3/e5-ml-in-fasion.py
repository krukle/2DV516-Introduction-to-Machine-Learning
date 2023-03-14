import os
import sys
import pandas   as pd
import numpy    as np
import globals  as gb
import seaborn  as sn
from os         import path
from matplotlib import pyplot as plt
# These are added since the windows store version of python (which im using) isn't properly importing the tensorflow-required dll files from path.
# os.add_dll_directory(path.join("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin"))
# os.add_dll_directory(path.join("C:/Program Files/NVIDIA/CUDNN/v 8.4.0.27/bin"))
# os.add_dll_directory(path.join("C:/Program Files/Zlib/dll_x64"))
import tensorflow as tf

LABEL_DICT = {
    0 : "T-shirt/top",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle boot"
}

data_train = pd.read_csv(path.join(gb.DATASETS_DIRECTORY, 'fashion-mnist_train.csv')).to_numpy()
data_test  = pd.read_csv(path.join(gb.DATASETS_DIRECTORY, 'fashion-mnist_test.csv')).to_numpy()

X_train, y_train = (data_train[:, 1:]/255.0).reshape(-1, 28, 28), data_train[:, 0]
X_test, y_test = (data_test[:, 1:]/255.0).reshape(-1, 28, 28), data_test[:, 0]

np.random.seed(7)
r = np.random.permutation(X_train.shape[0])

random_X = X_train[r]
random_y = y_train[r]
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(random_X[i])
    plt.axis("off")
    plt.title(LABEL_DICT[random_y[i]])
plt.show()

# Source: 
# https://www.tensorflow.org/tutorials/keras/classification
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting
# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
MODELS_DIRECTORIES = os.listdir('models')
accuracies = {}
if len(MODELS_DIRECTORIES) != 0:
    print("Running program using prerendered models.\n")
    models = {}
    for directory in MODELS_DIRECTORIES:
        try:
            models[directory] = tf.keras.models.load_model("models/" + directory)
        except:
            sys.exit(f"'models/{directory}' is not a valid model. Make sure there are only (valid) models in 'models/'.")
    for key, model in models.items():
        print(f"Model {key}:")
        accuracies[key] = model.evaluate(X_test, y_test, verbose=2)[1]
else:
    print("Rendering models and running program.\n")
    models = {
        "OG" :  tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ]),
        "L2" : tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                input_shape=(X_train.shape[0],)),
            tf.keras.layers.Dense(512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(10),
        ]),
        "Dropout" : tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='elu', input_shape=(X_train.shape[0],)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10),
        ]),
        "Combined" : tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, 
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                activation='elu', input_shape=(X_train.shape[0],)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, 
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, 
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, 
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10),
        ]),
        "CNN" : tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), padding='same', 
                activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', 
                activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', 
                activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
            tf.keras.layers.Dense(128, activation="relu"),
        ])
    }
    for key, model in models.items():   
        if key == "CNN":
            model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
        else:
            model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, verbose=False)
        model.save("models/" + key)
        accuracies[key] = model.evaluate(X_test, y_test, verbose=2)[1]
print(f"\nThe best model of the bunch is {max(accuracies, key=accuracies.get)} with an accuracy of {accuracies[max(accuracies, key=accuracies.get)]*100:.2f}%.")

prediction_arrays = models[max(accuracies, key=accuracies.get)].predict(X_test)
predictions = np.empty(y_test.shape)
for i in range(len(prediction_arrays)):
    predictions[i] = np.argmax(prediction_arrays[i])
sn.heatmap(tf.math.confusion_matrix(y_test, predictions), linewidths=.5, 
    annot=True, cmap="flare", xticklabels=LABEL_DICT.values(), 
    yticklabels=LABEL_DICT.values(), fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
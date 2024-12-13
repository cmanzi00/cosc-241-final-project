# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from model.neural_net import NeuralNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Set a parser for arguments passed via the terminal
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model after training")
args = vars(ap.parse_args())

# Set the learning rate, epochs count, and batch size
LR = 1e-3
EPOCHS = 24
BS = 128

# Load the MNIST dataset
print("---------- [INFO] ACCESSING MNIST DATASET ----------")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Pre-process the data

print("---------- [INFO] PRE-PROCESSING THE DATASET ----------")
# Reshape the dataset to have a channel dimension
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Normalize the data
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# Perform one-hot encoding on the labels
trainLabels = tf.one_hot(trainLabels, depth=10)
testLabels = tf.one_hot(testLabels, depth=10)

# Set the optimizer and loss function
print("---------- [INFO] BEGINNING MODEL COMPLILATION ----------")
opt = Adam(learning_rate=LR)
model = NeuralNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("---------- [INFO] BEGINNING NETWORK TRAINING ---------- ")

# Assuming `trainLabels` is your label array (TensorFlow tensor or numpy array)
trainLabels_np = trainLabels.numpy() if hasattr(
    trainLabels, "numpy") else trainLabels

# Flatten the labels to ensure they're 1D
trainLabels_flat = trainLabels_np.ravel()

# Compute class weights
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(trainLabels_flat),
    y=trainLabels_flat
)

# Convert to a dictionary
class_weights = {cls: weight for cls, weight in zip(
    np.unique(trainLabels_flat), class_weights_array)}

print("Class weights dictionary:", class_weights)

# Pass this dictionary to `model.fit`
model.fit(
    trainData,
    trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BS,
    epochs=EPOCHS,
)
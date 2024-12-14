# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from model.neural_net import NeuralNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
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
EPOCHS = 50
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

# Pass this dictionary to `model.fit`
model.fit(
    trainData,
    trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BS,
    epochs=EPOCHS,
)

# evaluate the network
print("---------- [INFO] MODEL EVALUATION ----------")
predictions = model.predict(testData)
print(classification_report(
    tf.argmax(testLabels, axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in range(10)]))

# Compute and visualize confusion matrix
y_true = tf.argmax(testLabels, axis=1)
y_pred = predictions.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=np.arange(10), yticklabels=np.arange(10))
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

# save model to the disk
print("---------- [INFO] SAVING MODEL ----------")
model.save(args["model"])

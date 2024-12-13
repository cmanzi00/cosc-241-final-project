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
    # class_weight=class_weights
)

# Define augmentation function


# def augment_image(image, label):
#     image = tf.image.random_flip_left_right(image)  # Random horizontal flip
#     image = tf.image.random_brightness(
#         image, max_delta=0.1)  # Random brightness
#     image = tf.image.random_contrast(
#         image, lower=0.8, upper=1.2)  # Random contrast
#     image = tf.image.random_crop(image, size=[26, 26, 1])  # Example crop
#     image = tf.image.resize(image, [28, 28])  # Resize back to original size
#     return image, label


# # Create a tf.data.Dataset pipeline
# dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
# dataset = dataset.map(augment_image)  # Apply augmentation
# dataset = dataset.batch(BS).repeat()

# dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
# dataset = dataset.batch(BS).repeat()

# # Create an ImageDataGenerator instance for augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     shear_range=0.1
# )

# # Fit the augmentation on training data
# datagen.fit(trainData)

# Update the model.fit call to use the augmented data
# datagen.flow(trainData, trainLabels, batch_size=BS, shuffle=True),
# model.fit(
#     dataset,
#     validation_data=(testData, testLabels),
#     steps_per_epoch=len(trainData) // BS,
#     epochs=EPOCHS
# )

# train the NN
# model.fit(
#     trainData, trainLabels,
#     validation_data=(testData, testLabels),
#     batch_size=BS,
#     epochs=EPOCHS)

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
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# save model to the disk
print("---------- [INFO] SAVING MODEL ----------")
model.save(args["model"])

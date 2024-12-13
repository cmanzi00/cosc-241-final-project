# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import L2


class NeuralNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # 1st CONV => RELU => POOL layers with kernel size 5x5 and default stride of 2
        model.add(Conv2D(32, (5, 5), padding="same",
                         input_shape=inputShape, activation="relu", kernel_regularizer=L2(0.001)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2nd CONV => RELU => POOL layers with kernel size 3x3 and default stride of 2
        model.add(Conv2D(64, (3, 3), padding="same",
                  activation="relu",
                  kernel_regularizer=L2(0.001)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 1st FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))

        # 2nd => RELU layers
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))

        # softmax classifier
        model.add(Dense(classes, activation="softmax"))

        # return the model
        return model

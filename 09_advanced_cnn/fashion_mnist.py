import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization, Flatten, Softmax

# read the data
train = pd.read_csv("./large_files/fashion_mnist/fashion-mnist_train.csv")
test = pd.read_csv("./large_files/fashion_mnist/fashion-mnist_test.csv")

# the data are already split ahead of time into the training and test set
# each of the 10 classes has the same number of samples in both the training and test set
# hence we only need to split the data into x and y
x_train, y_train = train.drop("label", axis=1), train.label
x_test, y_test = test.drop("label", axis=1), test.label

# reshape the x-s
x_train = np.array(x_train, dtype=np.float32).reshape([-1, 28, 28, 1]) / 255
x_test = np.array(x_test, dtype=np.float32).reshape([-1, 28, 28, 1]) / 255

# one-hot encode the y-s
y_train = np.array(pd.get_dummies(y_train), dtype=int)
y_test = np.array(pd.get_dummies(y_test), dtype=int)

# let's create a sequential keras model
model = Sequential()

# convolutional layer (3, 3) x 8
model.add(
    Conv2D(
        input_shape=(28, 28, 1),
        data_format="channels_last",
        kernel_size=(3, 3),
        filters=8,
        activation="relu"
    )
)

model.add(BatchNormalization())

# convolutional layer (3, 3) x 16
model.add(
    Conv2D(kernel_size=(3, 3), filters=16, activation="relu")
)

# maxpool
model.add(
    MaxPool2D(pool_size=(2, 2))
)

model.add(BatchNormalization())

# convolutional layer (3, 3) x 32
model.add(
    Conv2D(kernel_size=(3, 3), filters=32, activation="relu")
)

model.add(BatchNormalization())

# convolutional layer (3, 3) x 60
model.add(
    Conv2D(kernel_size=(3, 3), filters=64, activation="relu")
)

# maxpool
model.add(
    MaxPool2D(pool_size=(2, 2))
)

# flatten and add fully connected layers
model.add(Flatten())

model.add(BatchNormalization())

# dropout
model.add(Dropout(rate=0.2))

model.add(
    Dense(512, activation="relu")
)

model.add(BatchNormalization())

model.add(Dropout(rate=0.3))

model.add(
    Dense(256, activation="relu")
)

# add the final layer and apply the softmax function
model.add(
    Dense(10, activation="relu")
)

model.add(
    Softmax()
)

# compile the model
model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=["accuracy"])

# train the model
model.fit(
    x=x_train,
    y=y_train,
    epochs=5,
    validation_split=0.3
)

# check accuracy on the test set
preds = model.predict(x_test)
preds = np.argmax(preds, axis=1)
np.sum(preds == np.argmax(y_test, axis=1)) / len(preds)

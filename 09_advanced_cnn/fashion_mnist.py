import numpy as np
import pandas as pd
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPool2D, BatchNormalization, Flatten, Softmax

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

# let's create a keras model
i = Input(shape=(28, 28, 1))

# first convpool layer
x = Conv2D(kernel_size=(3, 3), filters=8)(i)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(kernel_size=(3, 3), filters=16)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = MaxPool2D(pool_size=(2, 2))(x)

# second convpool layer
x = Conv2D(kernel_size=(3, 3), filters=32)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(kernel_size=(3, 3), filters=64)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = MaxPool2D(pool_size=(2, 2))(x)

# flatten and add fully connected layers
x = Flatten()(x)

x = Dropout(rate=0.2)(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Dropout(rate=0.3)(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

# add the final layer and apply the softmax function
x = Dense(10)(x)
x = BatchNormalization()(x)
x = Activation("sigmoid")(x)
x = Softmax()(x)

# model
model = Model(inputs=i, outputs=x)

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

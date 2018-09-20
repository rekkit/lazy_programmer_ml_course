import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from dl_layers import resNetLayer, hiddenLayer, convolutionalLayer

# plotly
import plotly
import plotly.graph_objs as go

class resNetCNN(object):
    def __init__(self, img_shape, layers, n_classes):
        self.img_h, self.img_w, self.channels = img_shape
        self.layers = layers
        self.session = None
        self.train_costs = []
        self.validation_costs = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.initialized = False

        input_dimensions = (self.img_h, self.img_w, self.channels)
        for layer in self.layers:
            # initialize the layer by feeding it the dimensions of the input that will be passed
            layer.appendIn(input_dimensions)

            # get the input dimension for the next layer
            input_dimensions = layer.getOutputDimensions()

            # print the layer that was just initialized and it's output dimensions
            print("Layer: ", layer, "Output dimensions: ", input_dimensions)

        # add the fully connected layer at the end
        self.fc_layer = hiddenLayer(
            n_out=n_classes,
            layer_id=len(self.layers),
            activation_fn=tf.nn.softmax
        )

        self.fc_layer.appendIn(input_dimensions)

    def forwardLogits(self, x, is_training):
        z = x
        for layer in self.layers:
            # check if the forward function takes is_training as a parameter
            args = inspect.getfullargspec(layer.forward)[0]

            if "is_training" in args:
                z = layer.forward(z, is_training)
            else:
                z = layer.forward(z)

        # reshape output
        z = tf.reshape(z, shape=[-1, self.fc_layer.n_in])

        return self.fc_layer.forwardLogits(z)

    def forward(self, x, is_training):
        return tf.nn.softmax(
            self.forwardLogits(x, is_training)
        )

    def setSession(self, session):
        self.session = session

    def initializePlaceholders(self, x, y):
        self.tfX = tf.placeholder(dtype=tf.float32, shape=(None, *x.shape[1:]), name="tfX")
        self.tfT = tf.placeholder(dtype=tf.float32, shape=(None, y.shape[1]), name="tfT")

    def initialzeOperations(self):
        # since we're using batch normalization, we need to differentiate between logits when we're training (update
        # exponential moving average of the mean and variance of the batches) and testing (when we use the already
        # calculated values)
        self.logits_train = self.forwardLogits(x=self.tfX, is_training=True)
        self.logits_test = self.forwardLogits(x=self.tfX, is_training=False)

        # define the predict operation
        self.predict_op = tf.argmax(
            tf.nn.softmax(logits=self.logits_test),
            axis=1
        )

    def initializeCost(self):
        # similar to the logits, we need to differentiate between the test and train phases when defining the cost
        self.cost_train = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.tfT, logits=self.logits_train)
        )

        self.cost_test = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.tfT, logits=self.logits_test)
        )

    def initializeTrainingOperation(self, optimizer, clip_norm):
        if clip_norm is not None:
            gradients, variables = zip(*optimizer.compute_gradients(loss=self.cost_train))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_norm)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables))
        else:
            self.train_op = optimizer.minimize(loss=self.cost_train)

    def predict(self, x):
        return self.session.run(
            self.predict_op,
            feed_dict={self.tfX: x}
        )

    def fit(self, x, y, x_val, y_val, batch_size, n_epochs, session, optimizer=tf.train.AdamOptimizer, clip_norm=None,
            print_interval=20, new_round=False):

        # initialize if necessary
        if not self.initialized or new_round:
            # set the session
            self.setSession(session)

            # build the graph
            self.initializePlaceholders(x, y)
            self.initialzeOperations()
            self.initializeCost()
            self.initializeTrainingOperation(optimizer=optimizer, clip_norm=clip_norm)

            # initialize the variables
            self.session.run(tf.global_variables_initializer())

            # reset the lists that hold training metrics
            self.train_accuracy = []
            self.train_costs = []
            self.validation_accuracy = []
            self.validation_costs = []

            # set the initialized flag
            self.initialized = True

        # get the number of steps per epoch
        n_steps = x.shape[0] // batch_size

        # perform the training
        for i in range(n_epochs):
            x, y = shuffle(x, y)

            for j in range(n_steps):
                x_batch = x[j*batch_size: (j+1)*batch_size, ]
                y_batch = y[j*batch_size: (j+1)*batch_size, ]

                # perform the training step
                self.session.run(
                    self.train_op,
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                if j % print_interval == 0:
                    # calculate the training cost
                    self.train_costs.append(
                        self.session.run(
                            self.cost_test,
                            feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                        )
                    )

                    # calculate the validation cost
                    self.validation_costs.append(
                        self.session.run(
                            self.cost_test,
                            feed_dict={self.tfX: x_val, self.tfT: y_val}
                        )
                    )

                    # calculate the batch accuracy
                    self.train_accuracy.append(
                        np.mean(
                            self.session.run(self.predict_op, feed_dict={self.tfX: x_batch}) == np.argmax(y_batch, axis=1)
                        )
                    )

                    # calculate the training accuracy
                    self.validation_accuracy.append(
                        np.mean(
                            self.session.run(self.predict_op, feed_dict={self.tfX: x_val}) == np.argmax(y_val, axis=1)
                        )
                    )

                    print(
                        "Epoch: %d. Step: %d. Batch cost: %.2f. Batch accuracy: %.2f. Validation cost: %.2f. Validation accuracy: %.2f" % (
                            i, j, self.train_costs[-1], self.train_accuracy[-1], self.validation_costs[-1], self.validation_accuracy[-1]
                        )
                    )

    def plot_accuracy(self):
        g1 = go.Scatter(
            x=np.arange(len(self.train_accuracy)),
            y=self.train_accuracy,
            name="Train Accuracy"
        )

        g2 = go.Scatter(
            x=np.arange(len(self.train_accuracy)),
            y=self.validation_accuracy,
            name="Validadtion Accuracy"
        )

        layout = go.Layout(
            title="Accuracy",
            xaxis=dict(title="Step"),
            yaxis=dict(title="Accuracy")
        )

        figure = go.Figure(
            data=[g1, g2],
            layout=layout
        )

        plotly.offline.plot(figure)

    def plot_cost(self):
        g1 = go.Scatter(
            x=np.arange(len(self.train_costs)),
            y=self.train_costs,
            name="Train Accuracy"
        )

        g2 = go.Scatter(
            x=np.arange(len(self.validation_costs)),
            y=self.validation_costs,
            name="Validadtion Cost"
        )

        layout = go.Layout(
            title="Cost",
            xaxis=dict(title="Step"),
            yaxis=dict(title="Cost")
        )

        figure = go.Figure(
            data=[g1, g2],
            layout=layout
        )

        plotly.offline.plot(figure)


# data preprocessing
train_folders = glob("./large_files/fruits-360/Training/*")
test_folders = glob("./large_files/fruits-360/Test/*")

# decide on how many classes you want to train the model
train_on_n_classes = 10  # set to None if you want all classes

# training data
x = []
y = []
i = 0

for folder in tqdm(train_folders[:train_on_n_classes]):
    file_paths = glob(folder + "/*")

    for file_path in file_paths:
        x.append(imread(file_path))
        y.append(i)
    i += 1

x = np.array(x).reshape([-1, 100, 100, 3])
x = x / x.max()
y = np.array(
    pd.get_dummies(y)
)

# test data
x_val = []
y_val = []
i = 0

for folder in tqdm(test_folders[:train_on_n_classes]):
    file_paths = glob(folder + "/*")

    for file_path in file_paths:
        x_val.append(imread(file_path))
        y_val.append(i)
    i += 1

x_val = np.array(x_val).reshape([-1, 100, 100, 3])
x_val = x_val / x_val.max()
y_val = np.array(
    pd.get_dummies(y_val)
)

# MNIST
train = pd.read_csv("./large_files/mnist/train.csv")
test = pd.read_csv("./large_files/mnist/test.csv")

# transform data
x, x_val, y, y_val = train_test_split(
    train.drop("label", axis=1),
    train.label,
    stratify=train.label,
    test_size=0.3
)

# convert to numpy and reshape
x = np.array(x).reshape([-1, 28, 28, 1])
x_val = np.array(x_val).reshape([-1, 28, 28, 1])

y = np.array(
    pd.get_dummies(y)
).reshape([x.shape[0], -1])

y_val = np.array(
    pd.get_dummies(y_val)
).reshape([x_val.shape[0], -1])

# define the NN
cnn = resNetCNN(
    img_shape=x.shape[1:],
    layers=[
        convolutionalLayer(filter_h=5, filter_w=5, maps_out=8, layer_id=0),
        resNetLayer(filter_height=3, filter_width=3, layer_id=1),
        resNetLayer(filter_height=3, filter_width=3, layer_id=2),
        resNetLayer(filter_height=3, filter_width=3, layer_id=3, downsample=True),
        resNetLayer(filter_height=3, filter_width=3, layer_id=4),
        resNetLayer(filter_height=3, filter_width=3, layer_id=5),
        resNetLayer(filter_height=3, filter_width=3, layer_id=6, downsample=True),
        resNetLayer(filter_height=3, filter_width=3, layer_id=7),
        resNetLayer(filter_height=3, filter_width=3, layer_id=8),
        resNetLayer(filter_height=3, filter_width=3, layer_id=9, downsample=True),
        hiddenLayer(n_out=128, layer_id=0, activation_fn=tf.nn.relu)
    ],
    n_classes=y.shape[1]
)

# fit the data
cnn.fit(
    x=x,
    y=y,
    x_val=x_val,
    y_val=y_val,
    batch_size=128,
    n_epochs=10,
    session=tf.Session(),
    optimizer=tf.train.AdamOptimizer(learning_rate=0.000005),
    clip_norm=1
)

# plot metrics
cnn.plot_cost()
cnn.plot_accuracy()

# heatmap / confusiom matrix
preds = cnn.predict(x_val)
cm = confusion_matrix(y_true=np.argmax(y_val, axis=1), y_pred=preds)
f = [folder.split("\\")[-1] for folder in test_folders[:train_on_n_classes]]

g = go.Heatmap(z=cm)#, x=f, y=f)
plotly.offline.plot([g])

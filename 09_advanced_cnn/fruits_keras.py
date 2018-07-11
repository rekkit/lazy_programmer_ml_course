# keras imports
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.metrics import categorical_crossentropy
from keras.layers import Dense, Flatten, Activation
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# plotly import
import plotly
import plotly.graph_objs as go

# other libraries
import numpy as np
from glob import glob
from sklearn.metrics import confusion_matrix

# define the input image shape
IMAGE_SIZE = [100, 100]

# get the number of classes in the problem
n_classes = len(
    glob("./large_files/fruits-360/Training/*")
)

# set the batch size and the number of epochs
batch_size = 64
n_epochs = 10

# define where the training and test data can be found
train_folder_path = "./large_files/fruits-360/Training/"
test_folder_path = "./large_files/fruits-360/Test/"
train_paths = glob(train_folder_path + "*/*.jp*g")
test_paths = glob(test_folder_path + "*/*.jp*g")

# let's create the model
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)

# as a start, we don't want to train any of the layers
for layer in vgg.layers:
    layer.trainable = True

# flatten the output
x = Flatten()(vgg.output)

# predict
x = Dense(n_classes)(x)
preds = Activation("softmax")(x)

# create the model
model = Model(inputs=vgg.input, outputs=preds)

# compile the model
model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(lr=0.0001),
    metrics=["accuracy"]
)

# create an instance of a data generator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input  # VGG performs preprocessing to the data
)

# create generators for the training and test sets
train_generator = gen.flow_from_directory(
    directory=train_folder_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size
)

test_generator = gen.flow_from_directory(
    directory=test_folder_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size
)

# fit the model
r = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_paths) // batch_size,
    validation_data=test_generator,
    validation_steps=len(test_paths) // batch_size,
    epochs=n_epochs
)

# confusion matrix
predictions = []
labels = []
n_samples = len(test_paths)
i = 0

for x, y in gen.flow_from_directory(directory=test_folder_path, target_size=IMAGE_SIZE, shuffle=False):
    probs = model.predict(x)
    preds = np.argmax(probs, axis=1)
    y = np.argmax(y, axis=1)

    # append the new predictions
    predictions = np.append(predictions, preds)
    labels = np.append(labels, y)
    i += 1

    if i % 50 == 0:
        print(i)

    if i >= n_samples:
        break

cm = confusion_matrix(y_true=labels, y_pred=predictions)

# plot the confusion matrix
g = go.Heatmap(
    z=cm,
    x=list(test_generator.class_indices.keys()),
    y=list(test_generator.class_indices.keys())
)

layout = go.Layout(
    title="Fruits 360 Confusion Matrix",
    xaxis=dict(title="Ground Truth"),
    yaxis=dict(title="Predicted Values")
)

figure = go.Figure(
    data=[g],
    layout=layout
)

plotly.offline.plot(figure)


import numpy as np
import tensorflow as tf
from glob import glob
from dl_layers import resNetLayer
from sklearn.utils import shuffle
from skimage.io import imread

# training and test_paths
train_paths = glob("./large_files/fruits-360/Training/Apple*/*.jp*g")
test_paths = glob("./large_files/fruits-360/Test/Apple*/*.jp*g")

# placeholders
tfX = tf.placeholder(shape=(None, 100, 100, 3), dtype=tf.float32)
tfT = tf.placeholder(shape=(None, 100, 100, 3), dtype=tf.float32)

# define the layer
layer = resNetLayer(
    filter_height=3,
    filter_width=3,
    layer_id=0,
    activation_fn=tf.nn.relu
)

layer.appendIn((100, 100, 3))

# define the operations
predict_train = layer.forwardLogits(tfX, is_training=True)
residual = layer.forwardResidual(tfX, is_training=False)
cost = tf.reduce_sum(tf.square(predict_train - tfT))
train_op = tf.train.AdamOptimizer().minimize(cost)

# create session
sess = tf.Session()
init = tf.global_variables_initializer()

# define batch size and the number of epochs we want to do
batch_size = 50
n_epochs = 10

# initialize variables
sess.run(init)

for i in range(n_epochs):
    # shuffle the arrays
    train_paths = shuffle(train_paths)

    for j in range(len(train_paths) // batch_size):
        x_batch = []
        batch_paths = train_paths[j*batch_size: (j+1)*batch_size]

        for path in batch_paths:
            x_batch.append(imread(path))
        x_batch = np.array(x_batch) / 255

        _, c = sess.run(
            (train_op, cost),
            feed_dict={tfX: x_batch, tfT: x_batch}
        )

        print("Epoch: %d. Step: %d. Cost: %.3f" % (i, j, c))


t = sess.run(
    residual,
    feed_dict={tfX: imread(test_paths[0]).reshape([1, 100, 100, 3]) / 255}
)

# if this number is extremely small the skip connection is working properly
np.max(np.abs(t))

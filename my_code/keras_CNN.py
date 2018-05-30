import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras import backend as K

print(K.image_data_format())

data_dir = '/home/xiong/PycharmProjects/untitled/mnist'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_groundtruth = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)

with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

net = Conv2D(32, kernel_size=[5, 5], strides=[1, 1],
             activation='relu', padding='same',
             input_shape=[28, 28, 1])(x_image)
net = MaxPooling2D(pool_size=[2, 2])(net)
net = Conv2D(64, kernel_size=[5, 5], strides=[1, 1],
             activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=[2, 2])(net)
net = Flatten()(net)
net = Dense(1000, activation='relu')(net)
net = Dense(10, activation='softmax')(net)

from keras.objectives import categorical_crossentropy
cross_entropy = tf.reduce_mean(categorical_crossentropy(y_groundtruth, net))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
total_loss = cross_entropy + 7e-5 * l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

sess = tf.Session()

K.set_session(sess)

sess.run(tf.global_variables_initializer())

for step in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    lr = 0.01
    _, loss, l2_loss_value, total_loss_value = sess.run(
        [train_step, cross_entropy, l2_loss, total_loss],
        feed_dict={x: batch_xs, y_groundtruth: batch_ys, learning_rate: lr}
    )
    if (step + 1) % 1 == 0:
        print('step: %d, cross_entropy: %f, l2_loss: %f, total_loss: %f' %
              (step+1, loss, l2_loss_value, total_loss_value))
        correct_prediction = tf.equal(tf.argmax(y_groundtruth, 1), tf.argmax(net, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('train accuracy: %f' %
              sess.run(accuracy, feed_dict={x: batch_xs, y_groundtruth: batch_ys, learning_rate: lr}))
    if (step + 1) % 3 == 0:
        print('test accuracy: %f' %
              sess.run(accuracy, feed_dict={x: mnist.test.images, y_groundtruth: mnist.test.labels, learning_rate: lr}))

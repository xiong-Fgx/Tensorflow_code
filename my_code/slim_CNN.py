import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/home/xiong/PycharmProjects/untitled/mnist/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_groundtruth = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)

with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    out_conv1 = tf.contrib.slim.conv2d(x_image, 32, [5, 5],
                                       padding='SAME',
                                       activation_fn=tf.nn.relu)

with tf.name_scope('pool1'):
    out_pool1 = tf.contrib.slim.max_pool2d(out_conv1, [2, 2], stride=2,
                                           padding='VALID')

with tf.name_scope('conv2'):
    out_conv2 = tf.contrib.slim.conv2d(out_pool1, 64, [5, 5],
                                       padding='SAME',
                                       activation_fn=tf.nn.relu)

with tf.name_scope('pool2'):
    out_pool2 = tf.contrib.slim.max_pool2d(out_conv2, [2, 2], stride=2,
                                           padding='VALID')

with tf.name_scope('fc1'):
    fc1_flatten = tf.contrib.slim.avg_pool2d(out_pool2, out_pool2.shape[1:3], stride=[1, 1], padding='VALID')
    out_fc1 = tf.contrib.slim.conv2d(fc1_flatten, 1024, [1, 1], activation_fn=tf.nn.relu)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    out_dropout = tf.nn.dropout(out_fc1, keep_prob)

with tf.name_scope('fc2'):
    y = tf.squeeze(tf.contrib.slim.conv2d(out_dropout, 10, [1, 1], activation_fn=None))

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_groundtruth, logits=y)
)

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
total_loss = cross_entropy + 7e-5 * l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    lr = 0.1
    _, cross_entropy_value, l2_loss_value, total_loss_value = sess.run(
        [train_step, cross_entropy, l2_loss, total_loss],
        feed_dict={x: batch_xs, y_groundtruth: batch_ys, learning_rate: lr, keep_prob: 0.5}
    )
    if (step + 1) % 1 == 0:
        print('step: %d, cross_entropy: %f, l2_loss: %f, total_loss: %f' %
              (step + 1, cross_entropy_value, l2_loss_value, total_loss_value))
        correct_prediction = tf.equal(tf.argmax(y_groundtruth, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('train accuracy: %f' % sess.run(accuracy, feed_dict={x: batch_xs, y_groundtruth: batch_ys, keep_prob: 0.5, learning_rate: lr}))

    if(step + 1) % 300 == 0:
        print('test accuracy: %f' % sess.run(accuracy, feed_dict={x: mnist.test.images, y_groundtruth: mnist.test.labels, keep_prob: 0.5, learning_rate: lr}
                                             ))

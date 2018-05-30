import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/home/xiong/PycharmProjects/untitled/mnist'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_groundtruth = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)

with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1),
                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    l_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                           padding='SAME') + b_conv1
    out_conv1 = tf.nn.relu(l_conv1)

with tf.name_scope('pool1'):
    out_pool1 = tf.nn.max_pool(out_conv1, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='VALID')

with tf.name_scope('conv2'):
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),
                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    l_conv2 = tf.nn.conv2d(out_pool1, W_conv2, strides=[1, 1, 1, 1],
                           padding='SAME') + b_conv2
    out_conv2 = tf.nn.relu(l_conv2)

with tf.name_scope('pool2'):
    out_pool2 = tf.nn.max_pool(out_conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='VALID')

with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    out_pool2_flatten = tf.reshape(out_pool2, [-1, 7 * 7 * 64])
    l_fc1 = tf.matmul(out_pool2_flatten, W_fc1) + b_fc1
    out_fc1 = tf.nn.relu(l_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    out_dropout = tf.nn.dropout(out_fc1, keep_prob)

with tf.name_scope('fc2'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    l_fc2 = tf.matmul(out_dropout, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_groundtruth, logits=l_fc2))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection('WEIGHTS')])
total_loss = cross_entropy + 7e-5 * l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    lr = 0.01
    _, loss, l2_loss_value, total_loss_value = sess.run(
        [train_step, cross_entropy, l2_loss, total_loss],
        feed_dict={x: batch_xs, y_groundtruth: batch_ys, learning_rate: lr, keep_prob: 0.5}
    )

    if (step + 1) % 1 == 0:
        print('step %d, entropy_loss: %f, l2_loss: %f, total_loss: %f' %
              (step + 1, loss, l2_loss_value, total_loss_value))
        correct_prediction = tf.equal(tf.argmax(l_fc2, 1), tf.argmax(y_groundtruth, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('train accuracy: %f' % sess.run(accuracy,
                                              feed_dict={x: batch_xs, y_groundtruth: batch_ys, keep_prob: 0.5}
                                              ))
    if (step + 1) % 3 == 0:
        print('test accuracy: %f' % sess.run(accuracy,
                                             feed_dict={x: mnist.test.images, y_groundtruth: mnist.test.labels, keep_prob: 0.5}
                                             ))

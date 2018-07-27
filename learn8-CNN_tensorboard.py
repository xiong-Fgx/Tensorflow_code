import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope('stddev'): 
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    # x input tensor of shape [batch, in_weight, in_height, in_channels]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='label')
    with tf.name_scope('input'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    b_conv1 = bias_variable([32], name='b_conv1')
    
    conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(conv2d_1)

    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    b_conv2 = bias_variable([64], name='b_conv2')

    conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(conv2d_2)

    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')
    b_fc1 = weight_variable([1024], name='b_fc1')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')

    wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(wx_plus_b1)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10], name='W_fc2')
    b_fc2 = bias_variable([10], name='b_fc2')
    wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    prediction = tf.nn.softmax(wx_plus_b2)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                    name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('graph/mnist/train', sess.graph)
    test_writer = tf.summary.FileWriter('graph/mnist/test', sess.graph)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.6})
        train_writer.add_summary(summary, i)
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y:batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc, l = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print('test_accuracy:{0}, loss:{1}'.format(test_acc, l))
'''
本段代码相对于learn5的区别仅在于使用了keep_prob
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    x = tf.placeholder(dtype='float32', shape=[None, 784], name='input_x')
    y = tf.placeholder(dtype='float32', shape=[None, 10], name='labels')
    learning_rate = tf.Variable(0.01, dtype=tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(dtype='float32', name='keep_prob')

with tf.name_scope('layer'):
    with tf.name_scope('input_layer'):
        with tf.name_scope('W1'):
            W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.01), name='W1')
        with tf.name_scope('b1'):
            b1 = tf.Variable(tf.zeros([500]) + 0.01, name='b1')
        with tf.name_scope('L1'):
            L1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W1) + b1, name='L1_relu'), keep_prob)
    with tf.name_scope('hidden_layer'):
        with tf.name_scope('W2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.01), name='W2')
        with tf.name_scope('b2'):
            b2 = tf.Variable(tf.zeros([300]) + 0.01, name='b2')
        with tf.name_scope('L2'):
            L2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L1, W2) + b2, name='L2_relu'), keep_prob)
    with tf.name_scope('output_layer'):
        with tf.name_scope('W3'):
            W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.01), name='W3')
        with tf.name_scope('b3'):
            b3 = tf.Variable(tf.zeros([10]) + 0.01, name='b3')
        with tf.name_scope('out_logit'):
            out_logit = tf.nn.softmax(tf.matmul(L2, W3) + b3, name='logit')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_logit))
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(out_logit, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('graph/mnist', sess.graph)
    for i in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.6})
        writer.add_summary(summary, i)
        if(i % 2 == 0):
            acc, l = sess.run([accuracy, loss], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print("loss is:{0}, accuracy is:{1}".format(l, acc))

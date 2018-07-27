import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='W1')
b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='W2')
b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='W3')
b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
learning_rate = tf.Variable(0.01, dtype=tf.float32, name='learning_rate')

layer1_unactived = tf.matmul(x, W1) + b1 
layer1 = tf.nn.tanh(layer1_unactived, name='layer1')

layer2_unactived = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.tanh(layer2_unactived, name='layer2')

layer3_unactived = tf.matmul(layer2, W3) + b3
y_pred = tf.nn.softmax(layer3_unactived)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('graph/mnist', sess.graph)
    for i in range(201):
        sess.run(tf.assign(learning_rate, 0.01*(0.95**i)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        if(i % 2 == 0):
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print(acc)


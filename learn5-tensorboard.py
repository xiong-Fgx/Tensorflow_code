# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# batch_size = 100

# n_batch = mnist.train.num_examples // batch_size

# #将传进来的参数var求均值mean、标准差stddev、最大max最小值min和直方图histogram
# #使用tf.summary.scalar函数对上述几个统计量进行记录，同时记录参数var的直方图

# #使用tf.summary.scalar可以统计某个变量，在tensorboard上面展示，其实可以记录一些比较有意义的量，这里只是
# #将输入进来的参数进行计算一些比较常见的参数，比如均值方差等

# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram', var)

# with tf.name_scope('input'):
#     x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
#     y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_input')
#     learning_rate = tf.Variable(0.01, dtype=tf.float32, name='learning_rate')


# with tf.name_scope('layer'):
#     with tf.name_scope('input_layer'):
#         with tf.name_scope('W1'):
#             W1 = tf.Variable(tf.truncated_normal([784, 500],stddev=0.01), name='W1')
#             variable_summaries(W1)
#         with tf.name_scope('b1'):
#             b1 = tf.Variable(tf.zeros([500]) + 0.01, name='b1')
#             variable_summaries(b1)
#         with tf.name_scope('L1'):
#             L1 = tf.nn.tanh(tf.matmul(x, W1) + b1, name='L1')
#     with tf.name_scope('hidden_layer'):
#         with tf.name_scope('W2'):
#             W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.01), name='W2')
#             variable_summaries(W2)
#         with tf.name_scope('b2'):
#             b2 = tf.Variable(tf.zeros([300]) + 0.01, name='b2')
#             variable_summaries(b2)
#         with tf.name_scope('L2'):
#             L2 = tf.nn.relu(tf.matmul(L1, W2) + b2, name='L2')
#     with tf.name_scope('output_layer'):
#         with tf.name_scope('W3'):
#             W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.01), name='W3')
#             variable_summaries(W3)
#         with tf.name_scope('b3'):
#             b3 = tf.Variable(tf.zeros([10]) + 0.01, name='b3')
#             variable_summaries(b3)
#         prediction = tf.nn.softmax(tf.matmul(L2, W3) + b3, name = 'l3')

# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#     tf.summary.scalar('loss', loss)

# with tf.name_scope('optimizer'):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# with tf.name_scope('train'):
#     with tf.name_scope('correct_prediction'):
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    
#     with tf.name_scope('accuracy'):
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         tf.summary.scalar('accuracy', accuracy)

# init = tf.global_variables_initializer()
# #自动存入summary
# merged = tf.summary.merge_all()

# with tf.Session() as sess:
#     sess.run(init)
#     #定义所存图的目录
#     writer = tf.summary.FileWriter('graph/mnist', sess.graph)
#     for i in range(21):
#         #给tf.Variable变量赋值
#         sess.run(tf.assign(learning_rate, 0.01*(0.95**i)))
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             #sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
#             #summary=sess.run，需要通过这样一个步骤将得到summary，随后将summary记录到writer中，也就是writer.summary
#             summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys})
#         writer.add_summary(summary, i)
#         if(i % 2 == 0):
#             acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#             print(acc)
#     writer.close()

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100

n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)

with tf.name_scope('input'):
    with tf.name_scope('X'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_x')
    with tf.name_scope('y'):
        y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
    with tf.name_scope('learning_rate'):
        learning_rate = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')

with tf.name_scope('layer'):
    with tf.name_scope('input_layer'):
        with tf.name_scope('W1'):
            W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.01), name='W1')
        with tf.name_scope('b1'):
            b1 = tf.Variable(tf.zeros([500]) + 0.01, name='b1')
        with tf.name_scope('L1'):
            L1 = tf.nn.relu(tf.matmul(x, W1) + b1, name='L1')
    with tf.name_scope('hidden_layer'):
        with tf.name_scope('W2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.01), name='W2')
        with tf.name_scope('b2'):
            b2 = tf.Variable(tf.zeros([300]) + 0.01, name='b2')
        with tf.name_scope('L2'):
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2, name='L2')   
    with tf.name_scope('output'):
        with tf.name_scope('W3'):
            W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.01), name='W3')
        with tf.name_scope('b3'):
            b3 = tf.Variable(tf.zeros([10]) + 0.01, name='b3')

        L3 = tf.nn.softmax(tf.matmul(L2, W3) + b3, name='predict')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=L3), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(L3, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

global_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('graph/mnist', sess.graph)
    for i in range(21):
        #给tf.Variable变量赋值
        sess.run(tf.assign(learning_rate, 0.001*(0.95**i)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            #summary=sess.run，需要通过这样一个步骤将得到summary，随后将summary记录到writer中，也就是writer.summary
            summary, _ = sess.run([global_summary, optimizer], feed_dict={x: batch_xs, y: batch_ys})
        writer.add_summary(summary, i)
        if(i % 2 == 0):
            acc, l = sess.run([accuracy,loss], feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('accuracy: {0}, loss: {1}'.format(acc, l))
    writer.close()
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import time

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='intput_x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='labels')

x_image = tf.reshape(x, [-1, 28, 28, 1])
'''
变量说明：
1.
filter表示卷积核，比如：[5, 5, 1, 6]
前两个参数表示卷积核的尺寸大小为5×5，
第三个参数表示输入的通道数，
第四个参数表示输出的通道数，也就是卷积核的个数

2.
strides表示移动的步长，比如[1, 1, 1, 1]
表示在每个维度上移动的步长，
第一个参数表示batch
第二个三个参数表示height、width
第四个参数表示channel

3.
ksize表示池化矩阵的大小，比如[1, 2, 2, 1]
第一个参数表示batch
第二、三个参数为height和width
第四个参数表示channel

4.
padding表示补全方式，和卷积类似，可以取’VALID’ 或者’SAME’. 
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式. 
padding=’VALID’时，无自动填充。
padding=’SAME’时，自动填充，池化后保持shape不变。
'''
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
bias1 = tf.Variable(tf.truncated_normal([6]))
conv1 = tf.nn.conv2d(input=x_image, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.sigmoid(conv1 + bias1)

maxPool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
bias2 = tf.Variable(tf.truncated_normal([16]))
conv2 = tf.nn.conv2d(input=maxPool1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.sigmoid(conv2 + bias2)
maxPool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
bias3 = tf.Variable(tf.truncated_normal([120]))
conv3 = tf.nn.conv2d(input=maxPool2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3 = tf.nn.sigmoid(conv3 + bias3)


#全连接层
#权值和bias
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))
b_fc1 = tf.Variable(tf.truncated_normal([80]))

#将卷积的产出展开
h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#输出层
W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
y_output = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y * tf.log(y_output))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)

start_time = time.time()
for i in range(20000):
    batch_xs, batch_ys = mnist_data_set.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy, l = sess.run([accuracy, cross_entropy], feed_dict={x:batch_xs, y:batch_ys})
        print("step %d, training accuracy %g, loss:%g" % (i, train_accuracy, l))
        end_time = time.time()
        print("time: ", (end_time - start_time))
        start_time = end_time
    optimizer.run(feed_dict={x:batch_xs, y:batch_ys})

sess.close()
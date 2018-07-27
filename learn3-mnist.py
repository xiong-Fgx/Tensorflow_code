import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


batch_size = 100
#n_batch表示一个epoch需要用几个batch可以训练完
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x,W) + b)

loss = tf.reduce_mean(tf.square(y - prediction))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#这里进行准确率的预测步骤为：
#tf.cast:将变量类型变为指定的类型
#注意，tf.equal的输入变量是tensor，也就是类似矩阵的东西，因此得到的值也是一个tensor，并不只是一个值，而是很多个值在一起的
#在得到这个矩阵之后，使用tf.cast将T、F类型转成float，这样就可以用于reduce_mean来运算。运算时会对这个矩阵统计并得到结果。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))#此处得到的是一个boolean型的列表

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        if(epoch % 2 == 0):
            print("Iter {0}, test accuracy = {1}".format(epoch, acc))
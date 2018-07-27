import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
#多项式回归的预测模型为y=ax^3 + bx^2 + cx + d，也就是需要用到3个weight和1个bias一共四个参数
xs = np.linspace(-3, 3, 100)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, 100)

X = tf.placeholder(dtype=tf.float32, name='x')
Y = tf.placeholder(dtype=tf.float32, name='y')

W1 = tf.Variable(tf.random_normal([1]), name='W1')
W2 = tf.Variable(tf.random_normal([1]), name='W2')
W3 = tf.Variable(tf.random_normal([1]), name='W3')
B = tf.Variable(tf.random_normal([1]), name='B')

Y_pred = tf.multiply(W1, tf.pow(X,3)) + tf.multiply(W2, tf.pow(X, 2)) + tf.multiply(W3, X) + B

# 平方误差
loss = tf.square(Y_pred - Y, name='loss')

# 均值平方根
loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / 100
#另一种写法是reduce_mean()

learning_rate = 0.0001

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
        if(i % 5 == 0):
            print("emmm{0},{1}".format("sn", l))

    W1, W2, W3,B = sess.run([W1, W2, W3, B])

plt.scatter(xs, ys)
plt.plot(xs, W1*pow(xs,3) + W2*pow(xs,2) + W3*xs + B)
plt.show()

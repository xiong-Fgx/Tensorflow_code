#线性回归的TensorFlow实现，找到一条直线来拟合回归问题
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#数据点个数
n_observation = 100
#设置x和y
xs = np.linspace(-3,3,n_observation)
ys = np.sin(xs) + np.random.uniform(-0.5,0.5,n_observation)
#绘图
plt.scatter(xs,ys)
plt.show()


#设置x和y的placeholder，这两个值需要随后用数据填充，因此使用占位符将位置空出来，随后使用feed_dict的方法将数据填充进去即可
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

#Variable是需要学习的变量，因此使用Variable的方式来定义，如果是不可学习的量则使用constant来定义
W = tf.Variable(tf.random_normal([1]), name='weight')
B = tf.Variable(tf.random_normal([1]), name='bias')

#定义输出函数
Y_pred = tf.add(X * W, B)

#定义loss
loss = tf.square(Y - Y_pred, name='loss')

#学习率
learning_rate = 0.01

#定义优化器（优化器实例）
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#样本数量
n_sample = xs.shape[0]

#初始化各种变量
init = tf.global_variables_initializer()

#定义session
with tf.Session() as sess:
    #首先运行的是初始化程序
    sess.run(init)

    #进行50个epoch
    for i in range(50):
        total_loss = 0
        for x, y in zip(xs, ys):
            #可以这样理解吧，就是不管前面需要运行的有多少个，只要是在后面需要用到的，都要在这里feed进去。
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += 1
        if i % 5 == 0:
            print("epoch{0}:{1}".format(i, total_loss/n_sample))

    W, B = sess.run([W,B])

plt.scatter(xs, ys)
plt.plot(xs, xs*W+B)
plt.show()


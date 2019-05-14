import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
TEST_SIZE = 256


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # 三个卷积层与池化层，dropout一些神经元
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    # 全连接层，dropout一些神经元
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    # 输出层
    pyx = tf.matmul(l4, w_o)
    return pyx


def train(mnist):
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    trX = trX.reshape(-1, 28, 28, 1)
    teX = teX.reshape(-1, 28, 28, 1)
    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])

    # CNN结构：3个卷积层，3个池化层， 1个全连接层， 1个输出层
    # 设置卷积核大小为3*3
    w = init_weights([3, 3, 1, 32])
    w2 = init_weights([3, 3, 32, 64])
    w3 = init_weights([3, 3, 64, 128])
    w4 = init_weights([128 * 4 * 4, 625])  # 全连接层
    w_o = init_weights([625, 10])  # 输出层

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    predict_op = tf.argmax(py_x, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(100):
            training_batch = zip(range(0, len(trX), BATCH_SIZE),
                                 range(BATCH_SIZE, len(trX) + 1, BATCH_SIZE))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={
                    X: trX[start:end],
                    Y: trY[start:end],
                    p_keep_conv: 0.8,
                    p_keep_hidden: 0.5
                })
            # 获取测试batch
            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:TEST_SIZE]

            teY_ = sess.run(predict_op, feed_dict={
                X: teX[test_indices],
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0
            })
            correct_prediction = tf.equal(tf.argmax(teY[test_indices], axis=1), teY_)
            print(i, np.mean(sess.run(tf.cast(correct_prediction, tf.float32))))


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

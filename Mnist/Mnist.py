import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784 # 28*28
OUTPUT_NODE = 10

# 隐藏层节点数
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
# 正则化项在损失函数中的系数
REGULARIZAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

"""
加入一个隐藏层，一共是3层全连接神经网络
"""

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +  avg_class.average(biases1))
        return tf.matmul(layer1,  avg_class.average(weights2)) +  avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)
    global_step = tf.Variable(0, trainable=False)

    # 使用滑动平均可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有需要训练的参数上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 使用了滑动平均之后的计算结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,labels=tf.arg_max(y_, 1))
    # 当前batch 中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZAZTION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 训练过程中要更新神经网络参数，和参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g" %(i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g" % (TRAINING_STEPS, test_acc))

if __name__ == '__main__':
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)




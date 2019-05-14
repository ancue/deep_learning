from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

INPUT_NODE = 784  # 28*28
OUTPUT_NODE = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 100
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE = 0.05


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE])
    W = tf.Variable(tf.zeros([INPUT_NODE, OUTPUT_NODE]))
    b = tf.Variable(tf.zeros([OUTPUT_NODE]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for _ in range(TRAINING_STEPS):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            validate_feed = {x: mnist.validation.images,
                             y_: mnist.validation.labels}

            if _ % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d train step(s), validation accuracy is %g " % (_, validate_acc))

        test_acc = sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        })

        print(print("\nAfter %d train step(s), test accuracy is %g "
                    % (TRAINING_STEPS, test_acc)))


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)

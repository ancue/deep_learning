import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

lr = 0.001
training_iters = 100000
batch_size = 128

n_input = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
}

biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ])),
}

def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_input])

    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 采用LSTM循环网络单元
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        n_hidden_units, forget_bias=1.0, state_is_tuple=True
    )

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))
            step += 1



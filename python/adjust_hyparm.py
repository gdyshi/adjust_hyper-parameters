import argparse
# Copyright 2017 gdyshi. All Rights Reserved.
# github: https://github.com/gdyshi
# ==============================================================================

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import operator

FLAGS = None

layers = [784, 270, 90, 30, 10]
TRAINING_STEPS = 20000

learning_rate_up = 10
learning_rate_down = 0.00001
batch_size_up = 50
# batch_size_up = 1000
batch_size_down = 1
momentum_rate_up = 0.99999
momentum_rate_down = 0.9


def accuracy(y_pred, y_real):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_real, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc


def train(learning_rate, batch_size, momentum_rate):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print(batch_size)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y = inference(x)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.norm(y_ - y, axis=1) ** 2) / 2
    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate).minimize(loss, global_step=global_step)
    acc = accuracy(y, y_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if i % 1000 == 0:
                valid_acc = acc.eval(feed_dict={x: mnist.validation.images,
                                                y_: mnist.validation.labels})
                print("After %d training step(s), accuracy on validation is %g." % (i, valid_acc))
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        test_acc = acc.eval(feed_dict={x: mnist.test.images,
                                       y_: mnist.test.labels})
        print("After %d training step(s), accuracy on test is %g." % (TRAINING_STEPS, test_acc))


def generate_data(learning_rate_up, learning_rate_down, batch_size_up, batch_size_down, momentum_rate_up,
                  momentum_rate_down):
    learning_rates = np.logspace(np.log10(learning_rate_down), np.log10(learning_rate_up), num=10)
    batch_sizes = np.linspace(batch_size_down, batch_size_up, num=10).astype(int)
    momentum_rates = 1 - np.logspace(np.log10(1-momentum_rate_up), np.log10(1-momentum_rate_down), num=10)
    # learning_rates = np.random.random_integers(np.log10(learning_rate_down), high=np.log10(learning_rate_up), size=10)
    # batch_sizes = np.random.random_integers(batch_size_down, high=batch_size_up, size=10)
    # momentum_rates = np.random.random_integers(np.log10(momentum_rate_down), high=np.log10(momentum_rate_up), size=10)
    return learning_rates, batch_sizes, momentum_rates


def main(_):
    learning_rates, batch_sizes, momentum_rates = generate_data(learning_rate_up, learning_rate_down, batch_size_up,
                                                                batch_size_down, momentum_rate_up, momentum_rate_down)
    print('learning_rates:' + str(learning_rates))
    print('batch_sizes:' + str(batch_sizes))
    print('momentum_rates:' + str(momentum_rates))
    result = []
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for momentum_rate in momentum_rates:
                print('\n training on hy-parm:learning_rate:%g,batch_size:%d,momentum_rate:%g' % (
                    learning_rate, batch_size, momentum_rate))
                # acc = 0.9*momentum_rate
                acc = train(learning_rate, batch_size, momentum_rate)
                result.append({'learning_rate': learning_rate, 'batch_size': batch_size, 'momentum_rate': momentum_rate,
                               'accuracy': acc})
                # print('accuracy on test is %g' % (acc))
    result.sort(key=operator.itemgetter('accuracy'), reverse=True)
    for i in range(10):
        print('after loop the best %d learning_rate:%g batch_size:%d momentum_rate:%g' % (
        i + 1, result[i]['learning_rate'], result[i]['batch_size'], result[i]['momentum_rate']))
        print('accuracy on test is %g' % (result[i]['accuracy']))


def inference(x):
    for i in range(0, len(layers) - 1):
        X = x if i == 0 else y

        node_in = layers[i]
        node_out = layers[i + 1]
        W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') / (np.sqrt(node_in)))
        b = tf.Variable(np.random.randn(node_out).astype('float32'))
        z = tf.matmul(X, W) + b
        y = tf.nn.tanh(z)
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='E:\data\mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

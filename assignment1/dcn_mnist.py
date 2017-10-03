__author__ = 'tan_nguyen, Daniel LeJeune'

import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape, initializer=None):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    if initializer is None:
        W = tf.Variable(tf.truncated_normal(shape, 0.1))
    else:
        W = tf.Variable(lambda: initializer(shape=shape))

    return W


def bias_variable(shape, initializer=None):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    if initializer is None:
        b = tf.Variable(tf.constant(0.1, shape=shape))
    else:
        b = tf.Variable(lambda: initializer(shape=shape))

    return b


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_max


def main_2a():

    # Load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    jobid = '2a'

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Specify training parameters
            result_dir = './results_%s/' % jobid  # directory where the results from the training are saved
            max_step = 5500  # the maximum iterations. After max_step iterations, the training will stop no matter what

            start_time = time.time()  # start timing

            # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK

            # placeholders for input data and input labels
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])

            # reshape the input image
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            # first convolutional layer
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # second convolutional layer
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            # densely connected layer
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # softmax
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

            # setup training
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
            )
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Add a scalar summary for the snapshot loss.
            summary_op = tf.summary.scalar('CrossEntropy', cross_entropy)

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

            # Run the Op to initialize the variables.
            sess.run(init)

            # run the training
            for i in range(max_step):
                batch = mnist.train.next_batch(50)  # make the data batch, which is used in the training iteration.
                                                    # the batch size is 50
                if i % 100 == 0:
                    # output the training accuracy every 100 iterations
                    train_accuracy = sess.run(accuracy, feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))

                    # Update the events file which is used to monitor the training (in this case,
                    # only the training loss is monitored)
                    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

                # save the checkpoints every 1100 iterations
                if i % 1100 == 0 or i == max_step:
                    checkpoint_file = os.path.join(result_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=i)

                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

            # print test error
            print("test accuracy %g" % sess.run(accuracy, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

            stop_time = time.time()
            print('The training takes %f seconds to finish' % (stop_time - start_time))


def main_general(jobid, nonlinearity=tf.nn.relu, training_algorithm=None, W_initializer=None, b_initializer=None):

    # Load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Specify training parameters
            result_dir = './results_%s/' % jobid  # directory where the results from the training are saved
            max_step = 5500  # the maximum iterations. After max_step iterations, the training will stop no matter what

            start_time = time.time()  # start timing

            # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK

            # placeholders for input data and input labels
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])

            # reshape the input image
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            # first convolutional layer
            W_conv1 = weight_variable([5, 5, 1, 32], W_initializer)
            b_conv1 = bias_variable([32], b_initializer)
            h_conv1 = nonlinearity(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # second convolutional layer
            W_conv2 = weight_variable([5, 5, 32, 64], W_initializer)
            b_conv2 = bias_variable([64], b_initializer)
            h_conv2 = nonlinearity(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            # densely connected layer
            W_fc1 = weight_variable([7 * 7 * 64, 1024], W_initializer)
            b_fc1 = bias_variable([1024], b_initializer)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = nonlinearity(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # softmax
            W_fc2 = weight_variable([1024, 10], W_initializer)
            b_fc2 = bias_variable([10], b_initializer)
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

            # setup training
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
            )
            if training_algorithm is None:
                training_algorithm = tf.train.AdamOptimizer(1e-4)
            train_step = training_algorithm.minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            summaries = [tf.summary.scalar('CrossEntropy', cross_entropy)]

            for tensor, name in [
                (W_conv1, 'W_conv1'),
                (b_conv1, 'b_conv1'),
                (h_conv1, 'h_conv1'),
                (h_pool1, 'h_pool1'),
                (W_conv2, 'W_conv2'),
                (b_conv2, 'b_conv2'),
                (h_conv2, 'h_conv2'),
                (h_pool2, 'h_pool2'),
                (W_fc1, 'W_fc1'),
                (b_fc1, 'b_fc1'),
                (h_pool2_flat, 'h_pool2_flat'),
                (h_fc1, 'h_fc1'),
                (h_fc1_drop, 'h_fc1_drop'),
                (W_fc2, 'W_fc2'),
                (b_fc2, 'b_fc2'),
                (y_conv, 'y_conv')
            ]:
                tensor_flat = tf.reshape(tensor, [-1])
                summaries.append(tf.summary.scalar('%s_min' % name, tf.reduce_min(tensor_flat)))
                summaries.append(tf.summary.scalar('%s_max' % name, tf.reduce_max(tensor_flat)))
                mean, std = tf.nn.moments(tensor_flat, axes=[0], keep_dims=False)
                summaries.append(tf.summary.scalar('%s_mean' % name, mean))
                summaries.append(tf.summary.scalar('%s_std' % name, std))
                summaries.append(tf.summary.histogram('%s_hist' % name, tensor_flat))

            summary_op = tf.summary.merge(summaries)

            val_acc_summary = tf.summary.scalar('validation_accuracy', accuracy)
            test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

            # Run the Op to initialize the variables.
            sess.run(init)

            # run the training
            for i in range(max_step):
                batch = mnist.train.next_batch(50)  # make the data batch, which is used in the training iteration.
                                                    # the batch size is 50
                if i % 100 == 0:
                    # output the training accuracy every 100 iterations
                    train_accuracy = sess.run(accuracy, feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))

                    # Update the events file which is used to monitor the training (in this case,
                    # only the training loss is monitored)
                    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

                # save the checkpoints every 1100 iterations
                if i % 1100 == 0 or i == max_step:
                    summary_str_val = sess.run(val_acc_summary, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
                    summary_writer.add_summary(summary_str_val, i)
                    summary_str_test = sess.run(test_acc_summary, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                    summary_writer.add_summary(summary_str_test, i)
                    summary_writer.flush()
                    checkpoint_file = os.path.join(result_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=i)

                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

            # print test error
            print("test accuracy %g" % sess.run(accuracy, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

            stop_time = time.time()
            print('The training takes %f seconds to finish' % (stop_time - start_time))



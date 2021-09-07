"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

import tempfile
from resnets_utils import *
import tensorflow as tf
import numpy as np


class ResFi(object):

    def __init__(self, model_save_path='./model/model'):
        self.model_save_path = model_save_path


    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        """
        Implementation of the identity block as defined in Figure 3
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            W_conv1 = tf.Variable(tf.truncated_normal([1, 1, in_filter, f1], stddev=0.1))
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, f1, f2], stddev=0.1))
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            W_conv3 = tf.Variable(tf.truncated_normal([1, 1, f2, f3], stddev=0.1))
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #final step
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result


    def convolutional_block(self, X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 = tf.Variable(tf.truncated_normal([1, 1, in_filter, f1], stddev=0.1))
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, f1, f2], stddev=0.1))
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            W_conv3 = tf.Variable(tf.truncated_normal([1, 1, f2, f3], stddev=0.1))
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result

    def deepnn(self, x_input):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        Arguments:
        Returns:
        """
        #x = tf.pad(x_input, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "CONSTANT")
        with tf.variable_scope('reference'):
            training = tf.placeholder(tf.bool, name='training')

            # stage 1
            w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
            x = tf.nn.conv2d(x_input, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            # print('x=', x.shape)
            # assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

            # stage 2
            x = self.convolutional_block(x, 3, 64, [64, 64, 128], 2, 'a', training, stride=1)
            x = self.identity_block(x, 3, 128, [64, 64, 128], stage=2, block='b', training=training)
            x = self.identity_block(x, 3, 128, [64, 64, 128], stage=2, block='c', training=training)
            x11 = x
            x12 = x
            x12 = self.convolutional_block(x12, 3, 128, [64, 64, 128], 2, 'd', training)
            x12 = self.convolutional_block(x12, 3, 128, [64, 64, 128], 2, 'e', training)
            x12 = tf.image.resize_images(x12, [7, 7], method=0)
            x12 = tf.image.resize_images(x12, [15, 15], method=0)
            x12 = tf.nn.sigmoid(x12)
            x12 = x*x12
            gap1 = tf.nn.avg_pool(x, ksize=[1, 15, 15, 1], strides=[1, 1, 1, 1], padding='VALID', name='GAP1')
            fc = tf.contrib.layers.fully_connected(gap1, 32, activation_fn=tf.nn.relu)
            fc = tf.contrib.layers.fully_connected(fc, 128, activation_fn=tf.nn.sigmoid)
            x = tf.multiply(x, fc)
            x11 = tf.add(x11, x12)
            x = tf.concat([x, x11], 3)


            #stage 3
            x = self.convolutional_block(x, 3, 256, [128, 128, 256], 3, 'a', training)
            x = self.identity_block(x, 3, 256, [128, 128, 256], 3, 'b', training=training)
            x = self.identity_block(x, 3, 256, [128, 128, 256], 3, 'c', training=training)
            x = self.identity_block(x, 3, 256, [128, 128, 256], 3, 'd', training=training)
            x21 = x
            x22 = x
            x22 = self.convolutional_block(x22, 3, 256, [128, 128, 256], 3, 'e', training)
            x22 = tf.image.resize_images(x22, [8, 8], method=0)
            x22 = tf.nn.sigmoid(x22)
            x22 = x * x22
            gap1 = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='GAP2')
            fc = tf.contrib.layers.fully_connected(gap1, 64, activation_fn=tf.nn.relu)
            fc = tf.contrib.layers.fully_connected(fc, 256, activation_fn=tf.nn.sigmoid)
            x = tf.multiply(x, fc)
            x21 = tf.add(x21, x22)
            x = tf.concat([x, x21], 3)

            #stage 4
            x = self.convolutional_block(x, 3, 512, [256, 256, 512], 4, 'a', training)
            x = self.identity_block(x, 3, 512, [256, 256, 512], 4, 'b', training=training)
            x = self.identity_block(x, 3, 512, [256, 256, 512], 4, 'c', training=training)
            x = self.identity_block(x, 3, 512, [256, 256, 512], 4, 'd', training=training)
            x = self.identity_block(x, 3, 512, [256, 256, 512], 4, 'e', training=training)
            x = self.identity_block(x, 3, 512, [256, 256, 512], 4, 'f', training=training)
            x31 = x
            x32 = x
            x32 = self.convolutional_block(x32, 3, 512, [256, 256, 512], 4, 'g', training)
            x32 = tf.image.resize_images(x32, [4, 4], method=0)
            x32 = tf.nn.sigmoid(x32)
            x32 = x * x32
            gap1 = tf.nn.avg_pool(x, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID', name='GAP3')
            fc = tf.contrib.layers.fully_connected(gap1, 128, activation_fn=tf.nn.relu)
            fc = tf.contrib.layers.fully_connected(fc, 512, activation_fn=tf.nn.sigmoid)
            x = tf.multiply(x, fc)
            x31 = tf.add(x31, x32)
            x = tf.concat([x, x31], 3)

            #stage 5
            x = self.convolutional_block(x, 3, 1024, [512, 512, 1024], 5, 'a', training)
            x = self.identity_block(x, 3, 1024, [512, 512, 1024], 5, 'b', training=training)
            x = self.identity_block(x, 3, 1024, [512, 512, 1024], 5, 'c', training=training)
            x41 = x
            x42 = x
            x42 = self.convolutional_block(x42, 3, 1024, [512, 512, 1024], 5, 'e', training)
            x42 = tf.image.resize_images(x42, [2, 2], method=0)
            x42 = tf.nn.sigmoid(x42)
            x42 = x * x42
            gap1 = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='GAP4')
            fc = tf.contrib.layers.fully_connected(gap1, 256, activation_fn=tf.nn.relu)
            fc = tf.contrib.layers.fully_connected(fc, 1024, activation_fn=tf.nn.sigmoid)
            x = tf.multiply(x, fc)
            x41 = tf.add(x41, x42)
            x = tf.concat([x, x41], 3)
            print('x=', x.shape)

            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
            print("x=", x.shape)
            flatten = tf.layers.flatten(x)

            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=10, activation=tf.nn.softmax)

            #logits = tf.cast(logits, dtype=tf.int32, name=None)
        return logits, keep_prob, training

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
            tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
            # print('tv=', tv)
            regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=0.1)
            cross_entropy = cross_entropy + regularization_cost
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', cross_entropy_cost)
        return cross_entropy_cost

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_op = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy_op)
        return accuracy_op

    # merged = tf.summary.merge_all()

    def train(self, X_train, Y_train):
        features = tf.placeholder(tf.float32, [None, 30, 30, 3])
        labels = tf.placeholder(tf.int32, [None, 10])
        # lr = tf.placeholder(tf.int32)
        logits, keep_prob, train_mode = self.deepnn(features)
        cross_entropy = self.cost(logits, labels)
        accuracy = self.accuracy(logits, labels)

        print('logits=', logits)
        print('labels=', labels)


        with tf.name_scope('GradientDescentOptimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                 # train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001, momentum=0.1, decay=0.9).minimize(cross_entropy)
                 # train_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True).minimize(cross_entropy)
                 # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(cross_entropy)
                train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                 use_locking=False, name='Adam').minimize(cross_entropy)

        graph_location = r"C:\documents\PycharmProjects\ResNet_50\log"
        train_writer = tf.summary.FileWriter(graph_location, tf.get_default_graph())

        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            mincost = 300
            sess.run(tf.global_variables_initializer())
            # writer = tf.summary.FileWriter("log/", sess.graph) #写入到的位置
            for i in range(150000):
                X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
                # learning_rate = tf.train.piecewise_constant(i, boundaries=[40, 140625], values=[0.001, 0.0001, 0.00001])
                # learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=i, decay_steps=1000, decay_rate=0.96, staircase=True)
                #learning_rate = sess.run(learning_rate)

                train_step.run(feed_dict={features: X_mini_batch, labels: Y_mini_batch, keep_prob: 0.5, train_mode: True})
                if i % 20 == 0:
                    train_cost, acc = sess.run([cross_entropy, accuracy], feed_dict={features: X_mini_batch, labels:
                                                Y_mini_batch, keep_prob: 1.0, train_mode: False})
                    print('step %d, training cost %g, acc %g' % (i, train_cost, acc))
                    # print("lr=", learning_rate)
                    summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=train_cost)])
                    summary_ = tf.Summary(value=[tf.Summary.Value(tag="acc", simple_value=acc)])
                    train_writer.add_summary(summary, i)
                    train_writer.add_summary(summary_, i)
                    if train_cost < mincost:
                        mincost = train_cost
                        saver.save(sess, self.model_save_path)
        train_writer.close()
            # saver.save(sess, self.model_save_path)

    def test(self, test_features, test_labels, name='test'):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 30, 30, 3])
        y_ = tf.placeholder(tf.int64, [None, 10])

        logits, keep_prob, train_mode = self.deepnn(x)

        accuracy = self.accuracy(logits, y_)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_path)
            accu = sess.run(accuracy, feed_dict={x: test_features, y_: test_labels, keep_prob: 1.0, train_mode: False})
            print('%s accuracy %g' % (name, accu))


    def evaluate(self, train_features, train_labels, name='train'):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 30, 30, 3])
        y_ = tf.placeholder(tf.int64, [None, 10])

        logits, keep_prob, train_mode = self.deepnn(x)

        accuracy = self.accuracy(logits, y_)

        mini_batches = random_mini_batches(train_features, train_labels, mini_batch_size=1000, seed=None)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_path)
            sum = 0
            for i in range(45):
                x_mini_batch, y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
                accu = sess.run(accuracy, feed_dict={x: x_mini_batch, y_: y_mini_batch, keep_prob: 1.0, train_mode: False})
                sum = sum + accu
                print('%s accuracy %g' % (name, accu))
            print('sum=', sum)

def main(_):
    data_dir = r'C:\documents\PycharmProjects\ResNet_50\Data\augtrain'

    data = read_data(data_dir)
    #classes = len(set(labels))
    X_train, Y_train = process_orig_datasets(data)

    test_dir = r'C:\documents\PycharmProjects\ResNet_50\Data\test_'

    orig_data2 = read_data(test_dir)
    X_test, Y_test = process_orig_datasets(orig_data2)

    model = ResFi()
    model.train(X_train, Y_train)
    model.test(X_test, Y_test)
    model.evaluate(X_train, Y_train)


if __name__ == '__main__':
    tf.app.run(main=main)

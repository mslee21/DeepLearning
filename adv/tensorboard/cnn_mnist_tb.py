'''
CNN to classify MNIST handwritten digits
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

# Read in MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/mnist", one_hot=True)

LOGDIR='mnist_result/1'

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 1

# Network Parameters
n_input = 784 # input image shape = 28*28 grey scale
n_classes = 10 # 10 classes (0-9 digits)
dropout = 0.75 # probability to keep units during dropout

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="X")
y = tf.placeholder(tf.float32, [None, n_classes], name="Y")
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Wrappers
def reshape(x, xdim, ydim):
    return tf.reshape(x, shape=[-1, xdim, ydim, 1])

def conv2d(x, W, b, stride=1, name="Conv"):
#def conv2d(x, size_in, size_out, size_filter, stride=1, name="Conv"):
#    w = tf.Variable(tf.random_normal[size_filter, size_filter, size_in, size_out], name="W")
#    b = tf.Variable(tf.random_normal[size_out], name="B")
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME', name=name)
        #x = tf.nn.bias_add(x, b)
        act = tf.nn.relu(x + b)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def maxpool2d(x, size=2, stride=2, name="Pool"):
    with tf.name_scope(name):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = reshape(x, 28, 28)

    with tf.name_scope("Layer1"):
        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'], name="Conv1")
        print("Conv 1 = ", conv1)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, size=2, stride=2, name="Pool1")
        print("Conv 1 = ", conv1)

    with tf.name_scope("Layer2"):
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name="Conv2")
        print("Conv 2 = ", conv2)

        # Max Pooling (down-sampling) size=2, stride=2
        conv2 = maxpool2d(conv2, size=2, stride=2, name="Pool2")
        print("Conv 2 = ", conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    with tf.name_scope("FC"):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="WC1"),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="WC2"),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name="WD1"),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name="WOUT")
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name="BC1"),
    'bc2': tf.Variable(tf.random_normal([64]), name="BC2"),
    'bd1': tf.Variable(tf.random_normal([1024]), name="BD1"),
    'out': tf.Variable(tf.random_normal([n_classes], name="BOUT"))
}

# Construct model
with tf.name_scope("Model"):
    pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar("loss", loss)

# Evaluate model
with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    summ = tf.summary.merge_all()

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            l, acc, s = sess.run([loss, accuracy, summ], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            writer.add_summary(s, step)

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(l) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256], keep_prob: 1.}))

    writer.close()


import numpy as np
import tensorflow as tf
import cv2

import layer_def as ld

FLAGS = tf.app.flags.FLAGS

def fully_connected_model(x, keep_prob):
    # will not use keep prob at all for this model
    # create network
    # encodeing part first
    x = tf.reshape(x, [-1, 32*32])
    # fc1
    fc1 = ld.fc_layer(x, 1200, "encode_1")
    # fc2
    fc2 = ld.fc_layer(fc1, 1200, "encode_2")
    # y 
    y = ld.fc_layer(fc2, (FLAGS.hidden_size) * 2, "encode_3", False, True)
    mean, stddev = tf.split(1, 2, y)
    stddev =  tf.sqrt(tf.exp(stddev))
    # now decoding part
    # sample distrobution
    epsilon = tf.random_normal(mean.get_shape())
    y_sampled = mean + epsilon * stddev
    # fc4
    fc4 = ld.fc_layer(y_sampled, 1200, "decode_4")
    # fc5
    fc5 = ld.fc_layer(fc4, 1200, "decode_5")
    # fc6
    fc6 = ld.fc_layer(fc5, 1200, "decode_6")
    # fc7
    fc7 = ld.fc_layer(fc6, 32*32, "decode_7", False, True)
    x_prime = tf.nn.sigmoid(fc7)
    x_prime = tf.reshape(x_prime, [-1, 32, 32, 1])

    return mean, stddev, y_sampled, x_prime

def conv_model(x, keep_prob):
    # create network
    # encodeing part first
    # conv1
    conv1 = ld.conv_layer(x, 3, 2, 8, "encode_1")
    # conv2
    conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
    # conv3
    conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
    # conv4
    conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")
    # fc5 
    fc5 = ld.fc_layer(conv4, 128, "encode_5", True, False)
    # dropout maybe
    fc5_dropout = tf.nn.dropout(fc5, keep_prob)
    # y 
    y = ld.fc_layer(fc5_dropout, (FLAGS.hidden_size) * 2, "encode_6", False, True)
    mean, stddev = tf.split(1, 2, y)
    stddev =  tf.sqrt(tf.exp(stddev))
    # now decoding part
    # sample distrobution
    epsilon = tf.random_normal(mean.get_shape())
    y_sampled = mean + epsilon * stddev
    # fc7
    fc7 = ld.fc_layer(y_sampled, 128, "decode_7", False, False)
    # fc8
    fc8 = ld.fc_layer(fc7, 4*8*8, "decode_8", False, False)
    conv9 = tf.reshape(fc8, [-1, 8, 8, 4])
    # conv10
    conv10 = ld.transpose_conv_layer(conv9, 1, 1, 8, "decode_9")
    # conv11
    conv11 = ld.transpose_conv_layer(conv10, 3, 2, 8, "decode_10")
    # conv12
    conv12 = ld.transpose_conv_layer(conv11, 3, 1, 8, "decode_11")
    # conv13
    conv13 = ld.transpose_conv_layer(conv12, 3, 2, 1, "decode_12", True)
    # x_prime
    x_prime = conv13
    x_prime = tf.nn.sigmoid(x_prime)

    return mean, stddev, y_sampled, x_prime

def all_conv_model(x, keep_prob):
    # create network
    # encodeing part first
    # conv1
    conv1 = ld.conv_layer(x, 3, 2, 8, "encode_1")
    # conv2
    conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
    # conv3
    conv3 = ld.conv_layer(conv2, 3, 2, 16, "encode_3")
    # conv4 
    conv4 = ld.conv_layer(conv3, 3, 1, 8, "encode_4")
    # conv5 
    conv5 = ld.conv_layer(conv4, 3, 1, 8, "encode_5")
    # conv6
    conv6 = ld.conv_layer(conv5, 3, 2, FLAGS.hidden_size*2, "encode_6", True)
    mean_conv, stddev_conv = tf.split(3, 2, conv6)
    mean = tf.reshape(mean_conv, [-1, 4*4*FLAGS.hidden_size])
    stddev = tf.reshape(stddev_conv, [-1, 4*4*FLAGS.hidden_size])
    stddev =  tf.sqrt(tf.exp(stddev))
    # now decoding part
    # sample distrobution
    epsilon = tf.random_normal(mean.get_shape())
    y_sampled = mean + epsilon * stddev
    y_sampled_conv = tf.reshape(y_sampled, [-1, 4, 4, FLAGS.hidden_size])
    # conv7
    conv7 = ld.transpose_conv_layer(y_sampled_conv, 3, 2, 8, "decode_7")
    # conv8
    conv8 = ld.transpose_conv_layer(conv7, 3, 1, 8, "decode_8")
    # conv9
    conv9 = ld.transpose_conv_layer(conv8, 3, 1, 16, "decode_9")
    # conv10
    conv10 = ld.transpose_conv_layer(conv9, 3, 2, 8, "decode_10")
    # conv11
    conv11 = ld.transpose_conv_layer(conv10, 3, 1, 8, "decode_11")
    # conv12
    conv12 = ld.transpose_conv_layer(conv11, 3, 2, 1, "decode_12", True)
    # x_prime
    x_prime = conv12
    x_prime = tf.nn.sigmoid(x_prime)

    # reshape these just to make them look like other networks
    return mean, stddev, y_sampled, x_prime



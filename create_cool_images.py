

import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld

import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('hidden_size', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

def train(network_type):
  # set parameters
  if network_type == "num_balls_1_beta_0.1":
    FLAGS.num_balls=1
  elif network_type == "num_balls_1_beta_0.5":
    FLAGS.num_balls=1
  elif network_type == "num_balls_1_beta_1.0":
    FLAGS.num_balls=1

  """Eval net to get stddev."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 1])
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
    # y 
    y = ld.fc_layer(fc5, (FLAGS.hidden_size) * 2, "encode_6", False, True)
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

    # List of all Variables
    variables = tf.all_variables()

    # Load weights operator
    print('save file is ./checkpoints/train_store_' + network_type)
    ckpt = tf.train.get_checkpoint_state('./checkpoints/train_store_' + network_type)
    weight_saver = tf.train.Saver(variables)

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    weight_saver.restore(sess, ckpt.model_checkpoint_path)
    print("restored from" + ckpt.model_checkpoint_path)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)

    dat = b.bounce_vec(32, FLAGS.num_balls, FLAGS.batch_size)
    stddev_r = np.sum(sess.run([stddev],feed_dict={x:dat})[0], axis=0)/FLAGS.batch_size
    sample_y = sess.run([y_sampled],feed_dict={x:dat})[0]
    print(sample_y[0])

    # create grid
    y_p, x_p = np.mgrid[0:1:32j, 0:1:32j]
    
    for i in xrange(FLAGS.hidden_size-22):
      index = np.argmin(stddev_r)
      for j in xrange(5):
        z_f = np.copy(sample_y)
        z_f[0,index] = 1.5*j - 3.0
        print(sample_y[0])
        print(z_f[0])
        plt.subplot(FLAGS.hidden_size-22,5,j+5*i + 1)
        plt.pcolor(x_p, y_p, sess.run([x_prime],feed_dict={y_sampled:z_f})[0][0,:,:,0])
        plt.axis('off')
      # just make it big to get out of the way
      stddev_r[index] = 2.0
    plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  train("num_balls_1_beta_0.1")
  train("num_balls_1_beta_0.5")
  train("num_balls_1_beta_1.0")

  plt.figure(0)
  plt.plot(stddev_num_balls_1_beta_tenth, label="beta 0.1")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")


if __name__ == '__main__':
  tf.app.run()





import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import architecture as arc

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('hidden_size', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_string('model', 'conv',
                            """ either fully_connected, conv, or all_conv """)


def create_image(network_type):
  # set parameters
  if network_type in ("model_conv_num_balls_1_beta_0.1", "model_conv_num_balls_1_beta_0.5", "model_conv_num_balls_1_beta_1.0"):
    FLAGS.model="conv"
    FLAGS.num_balls=1
  if network_type in ("model_conv_num_balls_2_beta_0.1", "model_conv_num_balls_2_beta_0.5", "model_conv_num_balls_2_beta_1.0"):
    FLAGS.model="conv"
    FLAGS.num_balls=2
  elif network_type in ("model_fully_connected_num_balls_1_beta_0.1", "model_fully_connected_num_balls_1_beta_0.5", "model_fully_connected_num_balls_1_beta_1.0"):
    FLAGS.model="fully_connected"
    FLAGS.num_balls=1
    FLAGS.hidden_size=10
  elif network_type in ("model_fully_connected_num_balls_2_beta_0.1", "model_fully_connected_num_balls_2_beta_0.5", "model_fully_connected_num_balls_2_beta_1.0"):
    FLAGS.model="fully_connected"
    FLAGS.num_balls=2
    FLAGS.hidden_size=10
  elif network_type in ("model_all_conv_num_balls_1_beta_0.1", "model_all_conv_num_balls_1_beta_0.5", "model_all_conv_num_balls_1_beta_1.0"):
    FLAGS.model="all_conv"
    FLAGS.num_balls=1
    FLAGS.hidden_size=1
  elif network_type in ("model_all_conv_num_balls_2_beta_0.1", "model_all_conv_num_balls_2_beta_0.5", "model_all_conv_num_balls_2_beta_1.0"):
    FLAGS.model="all_conv"
    FLAGS.num_balls=2
    FLAGS.hidden_size=1

  """Eval net to get stddev."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 1])
 
    # no dropout on testing
    keep_prob = 1.0

    # make model
    if FLAGS.model=="fully_connected":
      mean, stddev, y_sampled, x_prime = arc.fully_connected_model(x, keep_prob)
    elif FLAGS.model=="conv":
      mean, stddev, y_sampled, x_prime = arc.conv_model(x, keep_prob)
    elif FLAGS.model=="all_conv":
      mean, stddev, y_sampled, x_prime = arc.all_conv_model(x, keep_prob)
    else:
      print("model requested not found, now some errors!")

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
    
    for i in xrange(10):
      index = np.argmin(stddev_r)
      for j in xrange(5):
        z_f = np.copy(sample_y)
        z_f[0,index] = 1.5*j - 3.0
        print(sample_y[0])
        print(z_f[0])
        plt.subplot(10,5,j+5*i + 1)
        plt.pcolor(x_p, y_p, sess.run([x_prime],feed_dict={y_sampled:z_f})[0][0,:,:,0])
        plt.axis('off')
      # just make it big to get out of the way
      stddev_r[index] = 2.0
    plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  create_image("model_conv_num_balls_1_beta_0.1")
  create_image("model_conv_num_balls_1_beta_0.5")
  create_image("model_conv_num_balls_1_beta_1.0")

  create_image("model_conv_num_balls_2_beta_0.1")
  create_image("model_conv_num_balls_2_beta_0.5")
  create_image("model_conv_num_balls_2_beta_1.0")

  create_image("model_fully_connected_num_balls_1_beta_0.1")
  create_image("model_fully_connected_num_balls_1_beta_0.5")
  create_image("model_fully_connected_num_balls_1_beta_1.0")

  create_image("model_fully_connected_num_balls_2_beta_0.1")
  create_image("model_fully_connected_num_balls_2_beta_0.5")
  create_image("model_fully_connected_num_balls_2_beta_1.0")

  create_image("model_all_conv_num_balls_1_beta_0.1")
  create_image("model_all_conv_num_balls_1_beta_0.5")
  create_image("model_all_conv_num_balls_1_beta_1.0")

  create_image("model_all_conv_num_balls_2_beta_0.1")
  create_image("model_all_conv_num_balls_2_beta_0.5")
  create_image("model_all_conv_num_balls_2_beta_1.0")

if __name__ == '__main__':
  tf.app.run()




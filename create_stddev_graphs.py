
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
tf.app.flags.DEFINE_integer('batch_size', 1000,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_string('model', 'conv',
                            """ either fully_connected, conv, or all_conv """)


def test_stddev(network_type):
  # set parameters (quick and dirty code just to make graphs fast)
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
      print("model requested not found, now some error!")

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
    stddev_r = np.sort(np.sum(sess.run([stddev],feed_dict={x:dat})[0], axis=0))
    return stddev_r/FLAGS.batch_size

def main(argv=None):  # pylint: disable=unused-argument
  stddev_model_conv_num_balls_1_beta_tenth = test_stddev("model_conv_num_balls_1_beta_0.1")
  stddev_model_conv_num_balls_1_beta_half = test_stddev("model_conv_num_balls_1_beta_0.5")
  stddev_model_conv_num_balls_1_beta_one = test_stddev("model_conv_num_balls_1_beta_1.0")

  stddev_model_conv_num_balls_2_beta_tenth = test_stddev("model_conv_num_balls_2_beta_0.1")
  stddev_model_conv_num_balls_2_beta_half = test_stddev("model_conv_num_balls_2_beta_0.5")
  stddev_model_conv_num_balls_2_beta_one = test_stddev("model_conv_num_balls_2_beta_1.0")

  stddev_model_fully_connected_num_balls_1_beta_tenth = test_stddev("model_fully_connected_num_balls_1_beta_0.1")
  stddev_model_fully_connected_num_balls_1_beta_half = test_stddev("model_fully_connected_num_balls_1_beta_0.5")
  stddev_model_fully_connected_num_balls_1_beta_one = test_stddev("model_fully_connected_num_balls_1_beta_1.0")

  stddev_model_fully_connected_num_balls_2_beta_tenth = test_stddev("model_fully_connected_num_balls_2_beta_0.1")
  stddev_model_fully_connected_num_balls_2_beta_half = test_stddev("model_fully_connected_num_balls_2_beta_0.5")
  stddev_model_fully_connected_num_balls_2_beta_one = test_stddev("model_fully_connected_num_balls_2_beta_1.0")

  stddev_model_all_conv_num_balls_1_beta_tenth = train("model_all_conv_num_balls_1_beta_0.1")
  stddev_model_all_conv_num_balls_1_beta_half = train("model_all_conv_num_balls_1_beta_0.5")
  stddev_model_all_conv_num_balls_1_beta_one = train("model_all_conv_num_balls_1_beta_1.0")

  stddev_model_all_conv_num_balls_2_beta_tenth = train("model_all_conv_num_balls_2_beta_0.1")
  stddev_model_all_conv_num_balls_2_beta_half = train("model_all_conv_num_balls_2_beta_0.5")
  stddev_model_all_conv_num_balls_2_beta_one = train("model_all_conv_num_balls_2_beta_1.0")

  plt.figure(0)
  plt.plot(stddev_model_conv_num_balls_1_beta_tenth, label="beta 0.1")
  plt.plot(stddev_model_conv_num_balls_1_beta_half, label="beta 0.5")
  plt.plot(stddev_model_conv_num_balls_1_beta_one, label="beta 1.0")
  plt.title("One Ball Dataset Conv")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")
  plt.savefig("one_ball_stddev_conv.png")

  plt.figure(1)
  plt.plot(stddev_model_conv_num_balls_2_beta_tenth, label="beta 0.1")
  plt.plot(stddev_model_conv_num_balls_2_beta_half, label="beta 0.5")
  plt.plot(stddev_model_conv_num_balls_2_beta_one, label="beta 1.0")
  plt.title("Two Ball Dataset Conv")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")
  plt.savefig("two_ball_stddev_conv.png")

  plt.figure(2)
  plt.plot(stddev_model_fully_connected_num_balls_1_beta_tenth, label="beta 0.1")
  plt.plot(stddev_model_fully_connected_num_balls_1_beta_half, label="beta 0.5")
  plt.plot(stddev_model_fully_connected_num_balls_1_beta_one, label="beta 1.0")
  plt.title("One Ball Dataset Fully Connected")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")
  plt.savefig("one_ball_stddev_fully_connected.png")

  plt.figure(3)
  plt.plot(stddev_model_fully_connected_num_balls_2_beta_tenth, label="beta 0.1")
  plt.plot(stddev_model_fully_connected_num_balls_2_beta_half, label="beta 0.5")
  plt.plot(stddev_model_fully_connected_num_balls_2_beta_one, label="beta 1.0")
  plt.title("Two Ball Dataset Fully Connected")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")
  plt.savefig("two_ball_stddev_fully_connected.png")

  plt.figure(4)
  plt.plot(stddev_model_all_conv_num_balls_1_beta_tenth, label="beta 0.1")
  plt.plot(stddev_model_all_conv_num_balls_1_beta_half, label="beta 0.5")
  plt.plot(stddev_model_all_conv_num_balls_1_beta_one, label="beta 1.0")
  plt.title("One Ball Dataset All Conv")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")
  plt.savefig("one_ball_stddev_all_conv.png")

  plt.figure(5)
  plt.plot(stddev_model_all_conv_num_balls_2_beta_tenth, label="beta 0.1")
  plt.plot(stddev_model_all_conv_num_balls_2_beta_half, label="beta 0.5")
  plt.plot(stddev_model_all_conv_num_balls_2_beta_one, label="beta 1.0")
  plt.title("Two Ball Dataset All Conv")
  plt.legend()
  plt.title("ordered standard deviation of latent encoding")
  plt.xlabel("latent variable number")
  plt.ylabel("average stddev")
  plt.savefig("two_ball_stddev_all_conv.png")

if __name__ == '__main__':
  tf.app.run()




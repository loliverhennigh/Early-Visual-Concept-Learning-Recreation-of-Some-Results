
import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import architecture as arc
import loss as ls  

import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('hidden_size', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size for testing""")
tf.app.flags.DEFINE_integer('num_runs', 100,
                            """number of batchs for generating loss""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_string('model', 'conv',
                            """ either fully_connected, conv, or all_conv """)
tf.app.flags.DEFINE_float('beta', .1,
                            """ beta constant """)
tf.app.flags.DEFINE_float('lr', .001,
                            """learning rate""")


def test_loss(network_type):
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

    # calc loss stuff
    loss_vae, loss_reconstruction, loss, train_op = ls.loss(mean, stddev, x, x_prime)

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

    loss_reconstruction_r = 0.0
    for step in xrange(FLAGS.num_runs):
      dat = b.bounce_vec(32, FLAGS.num_balls, FLAGS.batch_size)
      loss_reconstruction_r = loss_reconstruction_r + np.sum(sess.run([loss_reconstruction],feed_dict={x:dat})) / FLAGS.batch_size
      if np.sum(sess.run([loss_reconstruction],feed_dict={x:dat}))/ FLAGS.batch_size < 1.0:
        print(np.sum(sess.run([loss_reconstruction],feed_dict={x:dat}))/ FLAGS.batch_size)
    return loss_reconstruction_r/FLAGS.num_runs

def main(argv=None):  # pylint: disable=unused-argument
  loss_model_conv_num_balls_1 = np.zeros(3)
  loss_model_conv_num_balls_1[0] = test_loss("model_conv_num_balls_1_beta_0.1")
  loss_model_conv_num_balls_1[1] = test_loss("model_conv_num_balls_1_beta_0.5")
  loss_model_conv_num_balls_1[2] = test_loss("model_conv_num_balls_1_beta_1.0")

  loss_model_conv_num_balls_2 = np.zeros(3)
  loss_model_conv_num_balls_2[0] = test_loss("model_conv_num_balls_2_beta_0.1")
  loss_model_conv_num_balls_2[1] = test_loss("model_conv_num_balls_2_beta_0.5")
  loss_model_conv_num_balls_2[2] = test_loss("model_conv_num_balls_2_beta_1.0")

  loss_model_fully_connected_num_balls_1 = np.zeros(3)
  loss_model_fully_connected_num_balls_1[0] = test_loss("model_fully_connected_num_balls_1_beta_0.1")
  loss_model_fully_connected_num_balls_1[1] = test_loss("model_fully_connected_num_balls_1_beta_0.5")
  loss_model_fully_connected_num_balls_1[2] = test_loss("model_fully_connected_num_balls_1_beta_1.0")

  loss_model_fully_connected_num_balls_2 = np.zeros(3)
  loss_model_fully_connected_num_balls_2[0] = test_loss("model_fully_connected_num_balls_2_beta_0.1")
  loss_model_fully_connected_num_balls_2[1] = test_loss("model_fully_connected_num_balls_2_beta_0.5")
  loss_model_fully_connected_num_balls_2[2] = test_loss("model_fully_connected_num_balls_2_beta_1.0")

  loss_model_all_conv_num_balls_1 = np.zeros(3) 
  loss_model_all_conv_num_balls_1[0] = test_loss("model_all_conv_num_balls_1_beta_0.1")
  loss_model_all_conv_num_balls_1[1] = test_loss("model_all_conv_num_balls_1_beta_0.5")
  loss_model_all_conv_num_balls_1[2] = test_loss("model_all_conv_num_balls_1_beta_1.0")

  loss_model_all_conv_num_balls_2 = np.zeros(3) 
  loss_model_all_conv_num_balls_2[0] = test_loss("model_all_conv_num_balls_2_beta_0.1")
  loss_model_all_conv_num_balls_2[1] = test_loss("model_all_conv_num_balls_2_beta_0.5")
  loss_model_all_conv_num_balls_2[2] = test_loss("model_all_conv_num_balls_2_beta_1.0")

  beta = np.zeros(3)
  beta[0] = .1
  beta[1] = .5
  beta[2] = 1.0

  plt.figure(0)
  plt.scatter(beta, loss_model_conv_num_balls_1, label="conv", color='blue')
  plt.scatter(beta, loss_model_all_conv_num_balls_1, label="all_conv", color='red')
  plt.scatter(beta, loss_model_fully_connected_num_balls_1, label="fully_connected", color='green')
  plt.title("One Ball loss")
  plt.legend(loc=2)
  plt.xlabel("beta")
  plt.ylabel("loss for reconstruction")
  plt.savefig("figures/one_ball_reconstruction_loss.png")

  plt.figure(1)
  plt.scatter(beta, loss_model_conv_num_balls_2, label="conv", color='blue')
  plt.scatter(beta, loss_model_all_conv_num_balls_2, label="all_conv", color='red')
  plt.scatter(beta, loss_model_fully_connected_num_balls_2, label="fully_connected", color='green')
  plt.title("Two Ball loss")
  plt.legend(loc=2)
  plt.xlabel("beta")
  plt.ylabel("loss for reconstruction")
  plt.savefig("figures/two_ball_reconstruction_loss.png")

if __name__ == '__main__':
  tf.app.run()




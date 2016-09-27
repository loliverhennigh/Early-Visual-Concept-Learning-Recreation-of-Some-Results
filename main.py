

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

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('model', 'conv',
                            """ either fully_connected, conv, or all_conv """)
tf.app.flags.DEFINE_integer('hidden_size', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('max_step', 100000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """for dropout""")
tf.app.flags.DEFINE_float('beta', .1,
                            """ beta constant """)
tf.app.flags.DEFINE_float('lr', .001,
                            """learning rate""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")



# this will be the save file for the checkpoints
train_dir_save = FLAGS.train_dir + '_model_' + FLAGS.model + '_num_balls_' + str(FLAGS.num_balls) + '_beta_' + str(FLAGS.beta)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 1])

    # possible dropout inside (default is 1.0)
    keep_prob = tf.placeholder("float")

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

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(train_dir_save, graph_def=graph_def)

    for step in xrange(FLAGS.max_step):
      dat = b.bounce_vec(32, FLAGS.num_balls, FLAGS.batch_size)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t

      if step%2000 == 0:
        _ , loss_vae_r, loss_reconstruction_r, y_sampled_r, x_prime_r, stddev_r = sess.run([train_op, loss_vae, loss_reconstruction, y_sampled, x_prime, stddev],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("loss vae value at " + str(loss_vae_r))
        print("loss reconstruction value at " + str(loss_reconstruction_r))
        print("time per batch is " + str(elapsed))
        cv2.imwrite("real_balls.jpg", np.uint8(dat[0, :, :, :]*255))
        cv2.imwrite("generated_balls.jpg", np.uint8(x_prime_r[0, :, :, :]*255))
        stddev_r = np.sort(np.sum(stddev_r, axis=0))
        plt.plot(stddev_r/FLAGS.batch_size, label="step " + str(step))
        plt.legend()
        plt.savefig('stddev_num_balls_' + str(FLAGS.num_balls) + '_beta_' + str(FLAGS.beta) + '.png')
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(train_dir_save, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + train_dir_save)
        print("step " + str(step))

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(train_dir_save):
    tf.gfile.DeleteRecursively(train_dir_save)
  tf.gfile.MakeDirs(train_dir_save)
  train()

if __name__ == '__main__':
  tf.app.run()




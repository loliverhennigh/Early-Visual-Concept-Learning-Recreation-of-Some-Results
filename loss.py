
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def loss(mean, stddev, x, x_prime):
    # epsilon 
    epsilon = 1e-8
    # calc loss from vae
    kl_loss = 0.5 * (tf.square(mean) + tf.square(stddev) -
                         2.0 * tf.log(stddev + epsilon) - 1.0)
    loss_vae = FLAGS.beta * tf.reduce_sum(kl_loss)
    # log loss for reconstruction
    loss_reconstruction = tf.reduce_sum(-x * tf.log(x_prime + epsilon) -
                  (1.0 - x) * tf.log(1.0 - x_prime + epsilon)) 
    # save for tensorboard
    tf.scalar_summary('loss_vae', loss_vae)
    tf.scalar_summary('loss_reconstruction', loss_reconstruction)
    # calc total loss 
    loss = tf.reduce_sum(loss_vae + loss_reconstruction)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    return loss_vae, loss_reconstruction, loss, train_op

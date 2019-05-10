# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import numpy as np


from ops import *
from utils import *


class TRPOGAN(object):
  model_name = 'GAN'

  def __init__(
    self, sess, epoch, batch_size, z_dim, dataset_name,
    checkpoint_dir, result_dir, log_dir, data_dir
  ):
    self.sess = sess
    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.result_dir = result_dir
    self.log_dir = log_dir
    self.epoch = epoch
    self.batch_size = batch_size
    self.data_dir = data_dir

    if dataset_name in ['mnist', 'fashion-mnist']:
      # parameters
      self.input_height = 28
      self.input_weight = 28
      self.output_height = 28
      self.output_weight = 28
      
      self.z_dim = z_dim
      self.c_dim = 1

      # train
      self.learning_rate = 2e-4
      self.beta1 = 0.5

      # test
      self.sample_num = 64

      # load mnist
      self.data_X, self.data_y = load_mnist(self.data_dir)

      # get number of batches for a single epoch
      self.num_batches = len(self.data_X) // self.batch_size
    else:
      raise NotImplementedError

  def discrimintor(self, x, is_training=True, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
      net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
      net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
      net = tf.reshape(net, [self.batch_size, -1])
      net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
      out_logit = linear(net, 1, scope='d_fc4')
      out = tf.nn.sigmoid(out_logit)

      return out, out_logit, net

  def generator(self, z, reuse=False):
    

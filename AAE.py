#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

import prior_factory as prior

class AAE(object):
    model_name = "AAE"     # name for checkpoint

    def __init__(
        self, sess, epoch, batch_size, z_dim, dataset_name,
        checkpoint_dir, result_dir, log_dir, data_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_dir = data_dir

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.data_dir)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    # Gaussian Encoder
    def encoder(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            z = linear(net, self.z_dim, scope='en_fc4')

        return z

    # Bernoulli decoder
    def decoder(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
                   scope='de_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='de_dc4'))
            return out
    
    def discriminator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 128, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            logits = linear(net, 1, scope='de_fc2')
            out = tf.nn.sigmoid(logits)
        return out, logits

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """
        # encoding
        z = self.encoder(self.inputs, is_training=True, reuse=False)        


        # decoding
        out = self.decoder(z, is_training=True, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        D_real, D_real_logits = self.discriminator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits = self.discriminator(z, is_training=True, reuse=True)
        self.d_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        self.g_loss = -tf.reduce_mean(tf.log(D_fake))
        # d_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        # d_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        # self.d_loss = d_loss_real + d_loss_fake
        # self.g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)

        self.AAE_loss = self.neg_loglikelihood

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'encoder' in var.name]
        decoder_vars = [var for var in t_vars if 'decoder' in var.name]
        AAE_vars = g_vars + decoder_vars
        """ Training """
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.aae_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.AAE_loss, var_list=AAE_vars)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        loss_sum = tf.summary.scalar("loss", self.AAE_loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss = self.sess.run([self.aae_optim, self.merged_summary_op, self.AAE_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.sess.run(self.d_optim, feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.sess.run(self.g_optim, feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = prior.gaussian(self.batch_size, self.z_dim)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ learned manifold """
        if self.z_dim == 2:
            assert self.z_dim == 2

            z_tot = None
            id_tot = None
            for idx in range(0, 100):
                #randomly sampling
                id = np.random.randint(0,self.num_batches)
                batch_images = self.data_X[id * self.batch_size:(id + 1) * self.batch_size]
                batch_labels = self.data_y[id * self.batch_size:(id + 1) * self.batch_size]

                z = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})

                if idx == 0:
                    z_tot = z
                    id_tot = batch_labels
                else:
                    z_tot = np.concatenate((z_tot, z), axis=0)
                    id_tot = np.concatenate((id_tot, batch_labels), axis=0)

            save_scattered_image(z_tot, id_tot, -4, 4, name=check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_learned_manifold.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import scipy.io as sio
import keras.backend.tensorflow_backend as KTF

# Remove some unwanted warnings
import logging
logging.getLogger('tensorflow').disabled = True 

#Start Model
class Draw():
    def __init__(self):
 
        self.img_size = 32
        self.num_colors = 3

        self.attention_n = 5
        self.n_hidden = 256
        self.n_z = 10
        self.sequence_length = 10
        self.batch_size = 64
        self.epochs = 2
        self.share_parameters = False

        self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])

        self.e = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise

        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        self.cs = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        x = tf.reshape(self.images, [-1, self.img_size*self.img_size*self.num_colors])
        self.attn_params = []
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_size * self.img_size * self.num_colors)) if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            # r = self.read_basic(x,x_hat,h_dec_prev)
            r = self.read_attention(x,x_hat,h_dec_prev)
            # encode it to gauss distrib
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            # sample from the distrib to get z
            z = self.sampleQ(self.mu[t],self.sigma[t])
            # retrieve the hidden layer of RNN
            h_dec, dec_state = self.decode_layer(dec_state, z)
            # map from hidden layer -> image portion, and then write it.
            # self.cs[t] = c_prev + self.write_basic(h_dec)
            self.cs[t] = c_prev + self.write_attention(h_dec)
            h_dec_prev = h_dec

            #extract z_sample of final canvas # my edit
            if t == (self.sequence_length-1):
                self.z_sample = self.sampleQ(self.mu[t],self.sigma[t])

            self.share_parameters = True # from now on, share variables

        # the final timestep
        self.generated_images = tf.nn.sigmoid(self.cs[-1])

        # self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1-self.images) * tf.log(1e-10 + 1 - self.generated_images),1))
        self.generation_loss = tf.nn.l2_loss(x - self.generated_images)


        ## RECONSTRUCTION ERROR ##
        # recon_error to measure reconstruction error of epoch
        def recon_error(y_true, y_pred):
            return tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true))

        self.reconstruction_error = recon_error(x,self.generated_images)
        self.reconstruction_error = tf.reduce_mean(self.reconstruction_error)        


        kl_terms = [0]*self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - self.sequence_length*0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        self.cost = self.generation_loss + self.latent_loss
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v)
        self.train_op = optimizer.apply_gradients(grads)

        #Experimental gpu usage ####
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.config)
        KTF.set_session(self.sess)
        tf.Session(config=tf.ConfigProto(device_count = {'GPU': 1}))
        #Experimental gpu usage ####
        
        #self.sess = tf.Session() #uncomment for original
        self.sess.run(tf.initialize_all_variables())

    # given a hidden decoder layer:
    # locate where to put attention filters
    def attn_window(self, scope, h_dec):
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters,5,1)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.img_size+1)/2 * (gx_ + 1)
        gy = (self.img_size+1)/2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        delta = (self.img_size - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        # returns [Fx, Fy, gamma]

        self.attn_params.append([gx, gy, delta])

        return self.filterbank(gx,gy,sigma2,delta) + (tf.exp(log_gamma),)

    # Given a center, distance, and spread
    # Construct [attention_n x attention_n] patches of gaussian filters
    # represented by Fx = horizontal gaussian, Fy = vertical guassian
    def filterbank(self, gx, gy, sigma2, delta):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32),[1, -1])
        # centers for the individual patches
        mu_x = gx + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_y = gy + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im = tf.reshape(tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        Fy = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),1e-8)
        return Fx, Fy


    ### READ ###

    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x,x_hat], 1)

    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):
            # Fx,Fy = [64,5,32]
            # img = [64, 32*32*3]

            img = tf.reshape(img, [-1, self.img_size, self.img_size, self.num_colors])
            img_t = tf.transpose(img, perm=[3,0,1,2])

            # color1, color2, color3, color1, color2, color3, etc.
            batch_colors_array = tf.reshape(img_t, [self.num_colors * self.batch_size, self.img_size, self.img_size])
            Fx_array = tf.concat([Fx, Fx, Fx], 0)
            Fy_array = tf.concat([Fy, Fy, Fy], 0)

            Fxt = tf.transpose(Fx_array, perm=[0,2,1])

            # Apply the gaussian patches:
            glimpse = tf.matmul(Fy_array, tf.matmul(batch_colors_array, Fxt))
            glimpse = tf.reshape(glimpse, [self.num_colors, self.batch_size, self.attention_n, self.attention_n])
            glimpse = tf.transpose(glimpse, [1,2,3,0])
            glimpse = tf.reshape(glimpse, [self.batch_size, self.attention_n*self.attention_n*self.num_colors])
            # finally scale this glimpse w/ the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])
        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat([x, x_hat], 1)

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder",reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = dense(hidden_layer, self.n_hidden, self.n_z)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            logsigma = dense(hidden_layer, self.n_hidden, self.n_z)
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, next_state


    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latent, prev_state)

        return hidden_layer, next_state

    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = dense(hidden_layer, self.n_hidden, self.img_size*self.img_size*self.num_colors)
            # decoded_image_portion = tf.reshape(decoded_image_portion, [-1, self.img_size, self.img_size, self.num_colors])
        return decoded_image_portion

    def write_attention(self, hidden_layer):
        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n*self.attention_n*self.num_colors)

        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n, self.num_colors])
        w_t = tf.transpose(w, perm=[3,0,1,2])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)

        # color1, color2, color3, color1, color2, color3, etc.
        w_array = tf.reshape(w_t, [self.num_colors * self.batch_size, self.attention_n, self.attention_n])
        Fx_array = tf.concat([Fx, Fx, Fx], 0)
        Fy_array = tf.concat([Fy, Fy, Fy], 0)

        Fyt = tf.transpose(Fy_array, perm=[0,2,1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w_array, Fx_array))
        sep_colors = tf.reshape(wr, [self.batch_size, self.num_colors, self.img_size**2])
        wr = tf.reshape(wr, [self.num_colors, self.batch_size, self.img_size, self.img_size])
        wr = tf.transpose(wr, [1,2,3,0])
        wr = tf.reshape(wr, [self.batch_size, self.img_size * self.img_size * self.num_colors])
        return wr * tf.reshape(1.0/gamma, [-1, 1])
    
    
    
    def train(self):
        data = glob(os.path.join("C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\svhn_images", "*.jpg"))
        base = np.array([get_image(sample_file, 54, is_crop=False) for sample_file in data[0:64]])
        base += 1
        base /= 2

        ims("results/base.jpg",merge_color(base,[8,8]))

        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"\\training\\"))


        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters, '= total trainable parameters')


        cses = []
        gen_losses = []
        lat_losses = []
        reconstruct_errors = []
        

        for e in range(self.epochs):
            for i in range(int((len(data) / self.batch_size) - 2)):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file, 54, is_crop=False) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2
                cs, attn_params, gen_loss, lat_loss, _, reconstruct_error = self.sess.run([self.cs, self.attn_params, self.generation_loss, self.latent_loss, self.train_op, self.reconstruction_error], feed_dict={self.images: batch_images})

                if i%100==0:
                    print(("epoch %d iter %d genloss %f latloss %f -- Reconstruction_error: %f") % (e, i, gen_loss, lat_loss, reconstruct_error))                
                # print attn_params[0].shape
                # print attn_params[1].shape
                # print attn_params[2].shape
                if i % 800 == 0:

                    saver.save(self.sess, os.getcwd() + "/training/train", global_step=e*10000 + i)
                    print('Training progress saved, checkpoint made.')

                    cs = 1.0/(1.0+np.exp(-np.array(cs))) # x_recons=sigmoid(canvas)

                    for cs_iter in range(10):
                        results = cs[cs_iter]
                        results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_colors])
                        #print (results_square.shape)
                        ims("results/"+str(e)+"-"+str(i)+"-step-"+str(cs_iter)+".jpg",merge_color(results_square,[8,8]))
                        
            
                



    def view(self):

        data = glob(os.path.join("C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\svhn_images", "*.jpg"))

        base = np.array([get_image(sample_file, 54, is_crop=False) for sample_file in data[0:64]])
        base += 1
        base /= 2

        ims("results\\base.jpg",merge_color(base,[8,8]))

        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"\\training\\"))

        cs, attn_params, gen_loss, lat_loss = self.sess.run([self.cs, self.attn_params, self.generation_loss, self.latent_loss], feed_dict={self.images: base})
        print(("genloss %f latloss %f") % (gen_loss, lat_loss))

        cs = 1.0/(1.0+np.exp(-np.array(cs))) # x_recons=sigmoid(canvas)

        print (np.shape(cs))
        print (np.shape(attn_params))
            # cs[0][cent]

        for cs_iter in range(10):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_colors])

            print (np.shape(results_square))

            for i in range(64):
                center_x = int(attn_params[cs_iter][0][i][0])
                center_y = int(attn_params[cs_iter][1][i][0])
                distance = int(attn_params[cs_iter][2][i][0])

                size = 2;

                # for x in xrange(3):
                #     for y in xrange(3):
                #         nx = x - 1;
                #         ny = y - 1;
                #
                #         xpos = center_x + nx*distance
                #         ypos = center_y + ny*distance
                #
                #         xpos2 = min(max(0, xpos + size), 63)
                #         ypos2 = min(max(0, ypos + size), 63)
                #
                #         xpos = min(max(0, xpos), 63)
                #         ypos = min(max(0, ypos), 63)
                #
                #         results_square[i,xpos:xpos2,ypos:ypos2,0] = 0;
                #         results_square[i,xpos:xpos2,ypos:ypos2,1] = 1;
                #         results_square[i,xpos:xpos2,ypos:ypos2,2] = 0;
                # print "%f , %f" % (center_x, center_y)

            print (results_square)

            ims("results/view-clean-step-"+str(cs_iter)+".jpg",merge_color(results_square,[8,8]))


data1 = glob(os.path.join("C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\svhn_images", "*.jpg"))

model = Draw()
model.train()
#model.view()
#model.is_outlier(data1)

del data1


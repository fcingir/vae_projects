#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
	python draw_GRAY_automated_fashion.py --data_dir=/tmp/draw --read_attn=True --write_attn=True
	python draw_GRAY_automated_fashion.py --data_dir=/tmp/draw --read_attn=False --write_attn=False

Author: Eric Jang
"""

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import struct

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

KTF.set_session(sess)

tf.Session(config=tf.ConfigProto(device_count = {'GPU': 1}))

###

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
FLAGS = tf.flags.FLAGS


def DRAW(train_data, dim_representation, nodes):
    tf.reset_default_graph() # to loop over

    ## MODEL PARAMETERS ## 

    A,B = 28,28 # image width,height
    img_size = B*A # the canvas size
    enc_size = nodes # number of hidden units / output size in LSTM
    dec_size = nodes
    read_n = 5 # read glimpse grid width/height
    write_n = 5 # write glimpse grid width/height
    read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
    write_size = write_n*write_n if FLAGS.write_attn else img_size
    z_size=dim_representation # QSampler output size
    T=10 # MNIST generation sequence length
    batch_size=64 # training minibatch size, 64 is in original code
    train_iters=5000
    epochs = 100
    learning_rate=1e-3 # learning rate for optimizer
    eps=1e-8 # epsilon for numerical stability

    ## BUILD MODEL ## 

    DO_SHARE=None # workaround for variable_scope(reuse=True)

    #batch_X = np.reshape(batch_X, (-1, 784)) # Fatty experiment
    x = tf.placeholder(tf.float32,shape=(None,img_size)) # input (batch_size * img_size)
    e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
    lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
    lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True) # decoder Op

    def linear(x,output_dim):
        """
        affine transformation Wx+b
        assumes x.shape = (batch_size, num_features)
        """
        w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
        b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x,w)+b

    def filterbank(gx, gy, sigma2,delta, N):
        grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
        a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
        Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
        # normalize, sum over A and B dims
        Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
        Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
        return Fx,Fy

    def attn_window(scope,h_dec,N):
        with tf.variable_scope(scope,reuse=DO_SHARE):
            params=linear(h_dec,5)
        # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
        gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params,5,1)
        gx=(A+1)/2*(gx_+1)
        gy=(B+1)/2*(gy_+1)
        sigma2=tf.exp(log_sigma2)
        delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
        return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

    ## READ ## 
    def read_no_attn(x,x_hat,h_dec_prev):
        return tf.concat([x,x_hat], 1)

    def read_attn(x,x_hat,h_dec_prev):
        Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
        def filter_img(img,Fx,Fy,gamma,N):
            Fxt=tf.transpose(Fx,perm=[0,2,1])
            img=tf.reshape(img,[-1,B,A])
            glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
            glimpse=tf.reshape(glimpse,[-1,N*N])
            return glimpse*tf.reshape(gamma,[-1,1])
        x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
        x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
        return tf.concat([x,x_hat], 1) # concat along feature axis

    read = read_attn if FLAGS.read_attn else read_no_attn

    ## ENCODE ## 
    def encode(state,input):
        """
        run LSTM
        state = previous encoder state
        input = cat(read,h_dec_prev)
        returns: (output, new_state)
        """
        with tf.variable_scope("encoder",reuse=DO_SHARE):
            return lstm_enc(input,state)

    ## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

    def sampleQ(h_enc):
        """
        Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
        mu is (batch,z_size)
        """
        with tf.variable_scope("mu",reuse=DO_SHARE):
            mu=linear(h_enc,z_size)
        with tf.variable_scope("sigma",reuse=DO_SHARE):
            logsigma=linear(h_enc,z_size)
            sigma=tf.exp(logsigma)
        return (mu + sigma*e, mu, logsigma, sigma)

    ## DECODER ## 
    def decode(state,input):
        with tf.variable_scope("decoder",reuse=DO_SHARE):
            return lstm_dec(input, state)

    ## WRITER ## 
    def write_no_attn(h_dec):
        with tf.variable_scope("write",reuse=DO_SHARE):
            return linear(h_dec,img_size)

    def write_attn(h_dec):
        with tf.variable_scope("writeW",reuse=DO_SHARE):
            w=linear(h_dec,write_size) # batch x (write_n*write_n)
        N=write_n
        w=tf.reshape(w,[batch_size,N,N])
        Fx,Fy,gamma=attn_window("write",h_dec,write_n)
        Fyt=tf.transpose(Fy,perm=[0,2,1])
        wr=tf.matmul(Fyt,tf.matmul(w,Fx))
        wr=tf.reshape(wr,[batch_size,B*A])
        #gamma=tf.tile(gamma,[1,B*A])
        return wr*tf.reshape(1.0/gamma,[-1,1])

    write=write_attn if FLAGS.write_attn else write_no_attn

    ## STATE VARIABLES ## 

    cs=[0]*T # sequence of canvases
    mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
    # initial states
    h_dec_prev=tf.zeros((batch_size,dec_size))
    enc_state=lstm_enc.zero_state(batch_size, tf.float32)
    dec_state=lstm_dec.zero_state(batch_size, tf.float32)

    ## DRAW MODEL ## 

    # construct the unrolled computational graph
    for t in range(T):
        c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
        x_hat=x-tf.sigmoid(c_prev) # error image
        r=read(x,x_hat,h_dec_prev)
        h_enc,enc_state=encode(enc_state,tf.concat([r,h_dec_prev], 1)) # Encode to prepare for construction of probabilities
        z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc)                 # probabilities sampling / Latent Space
        h_dec,dec_state=decode(dec_state,z)                            # Decode / Sample from Z's
        cs[t]=c_prev+write(h_dec) # store results
        h_dec_prev=h_dec
        DO_SHARE=True # from now on, share variables

    ## LOSS FUNCTION ## 

    def binary_crossentropy(t,o):
        return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

    ## RECONSTRUCTION ERROR ##
    # recon_error to measure reconstruction error of epoch
    def recon_error(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true))

    # reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
    x_recons=tf.nn.sigmoid(cs[-1])

    # after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
    Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
    Lx=tf.reduce_mean(Lx)

    reconstruction_error = recon_error(x,x_recons)
    reconstruction_error = tf.reduce_mean(reconstruction_error)

    kl_terms=[0]*T
    for t in range(T):
        mu2=tf.square(mus[t])
        sigma2=tf.square(sigmas[t])
        logsigma=logsigmas[t]
        kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-.5 # each kl term is (1xminibatch)
    KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
    Lz=tf.reduce_mean(KL) # average over minibatches

    cost=Lx+Lz

    ## OPTIMIZER ## 

    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(cost)
    for i,(g,v) in enumerate(grads):
        if g is not None:
            grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)

    ## RUN TRAINING ## 

    fetches=[]
    fetches.extend([Lx,Lz,train_op,reconstruction_error])
    Lxs=[0]*train_iters
    Lzs=[0]*train_iters
    reconstruction_errors =[0]*train_iters ##### Test


    sess=tf.InteractiveSession()

    saver = tf.train.Saver() # saves variables learned during training
    tf.global_variables_initializer().run()
    #saver.restore(sess, "/tmp/draw/drawmodel_fashion_ndim10_256.ckpt") # to restore from model, uncomment this line

    
    train_datasize = train_data.shape[0]

    period = int(train_datasize/batch_size) #rounds down
    for e in range(epochs):
            #tf.compat.v1.enable_eager_execution() # enable learning for training set
            print('Epoch' + str(e+1))
            idxs = np.random.permutation(train_datasize) #shuffled ordering
            x_random = train_data[idxs]

            for i in range(period):
                    batch_xtrain = x_random[i * batch_size:(i+1) * batch_size] # Fatty experiment

                    feed_dict={x:batch_xtrain}
                    results=sess.run(fetches,feed_dict)
                    Lxs[i],Lzs[i],_, reconstruction_errors[i]=results
                    if i%100==0:
                            print("iter=%d : TRAINING Lx: %f Lz: %f Recon_error: %f" % (i,Lxs[i],Lzs[i], reconstruction_errors[i]))

    ## TRAINING FINISHED ##


    canvases=sess.run(cs,feed_dict) # generate some examples
    canvases=np.array(canvases) # T x batch x img_size

    #out_file=os.path.join(FLAGS.data_dir,"draw_data_fashion_ndim10_256.npy")
    #np.save(out_file,[canvases,Lxs,Lzs])
    #print("Outputs saved in file: %s" % out_file)

    ckpt_file=os.path.join(os.getcwd(),"drawmodel_fashion_ndim"+str(dim_representation)+"_"+str(nodes)+".ckpt")
    print("Model saved in file: %s" % saver.save(sess,ckpt_file))

    sess.close()
    
    return 


#Model training block

from dataset_unpacking_utility import prepare_dataset_fashion_mnist, prepare_dataset_mnist

dataset_prep = np.asarray([prepare_dataset_fashion_mnist]) #1 model takes about 85 minutes to train if you want 100 epochs. So cycle through them one by one. 


def train_all_datasets(add_list_of_datasets):
    #create names for weights_test
    weight_names_list = []
    for i in range(len(dataset_prep)):
        filename = 'dataset_'+str(i)+'_DRAW_fashion'
        weight_names_list.append(filename)
    weight_names_list = np.array(weight_names_list)
    
    return_here_path = os.getcwd()
    for i in range(len(dataset_prep)):
        os.chdir(return_here_path)
        
        X_train, _, _, _ = dataset_prep[i]() #cycle through each function to create dataset
        X_train = np.reshape(X_train, (X_train.shape[0], 784))
        weights_name_dataset = weight_names_list[i] #create basename for folder to be created for a particular dataset
        
        model_hyperparameters = [#[2,256],[10,256],[20,256],[100,256],
                         #[2,512],[10,512],[20,512],[100,512],
                         #[2,X_train.shape[1]],[10,X_train.shape[1]],[20,X_train.shape[1]],
                         [100,X_train.shape[1]]]
        
        #automate folder creation for each dataset
        path_dataset = os.path.abspath(os.getcwd())+'\\'+str(weights_name_dataset)
        if not os.path.exists(path_dataset):
            os.makedirs(path_dataset)
            
        
        for j in model_hyperparameters:
            os.chdir(path_dataset) 
            weights_name = weights_name_dataset+'_ndim_'+str(j[0])+'_filters_'+str(j[1])
            DRAW(X_train, dim_representation = j[0], nodes = j[1]) #(train_set, test_set, modelname, desired epochs, filename to save to)
            
    os.chdir(return_here_path) #return to working directory of this script
    return
train_all_datasets(dataset_prep)
    
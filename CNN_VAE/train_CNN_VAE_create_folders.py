## Basic necessities to run tensorflow on my system

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

KTF.set_session(sess)

tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count = {'GPU': 1}))

import keras
keras.__version__

##

import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Reshape, Dropout, Flatten, Activation, Concatenate
from tensorflow.keras import losses
from tensorflow.keras import initializers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D 
from tensorflow.keras import regularizers

from keras.models import load_model

## IMPORTS DONE ##


def train_model(X_train, X_test, epochs, dim_representation, b_f, save_weight_name=str):
    
    # vae_loss function to perform stochastic gradient descent on
    def vae_loss(y_true,y_pred):

        recontruction_loss = K.mean(K.binary_crossentropy(y_pred, y_true))
        latent_loss = -0.5 * K.mean(1 + z_std_sq_log - K.square(z_mean) - K.exp(z_std_sq_log), axis=-1 )
        return recontruction_loss + 1e-8*latent_loss

    # recon_error to measure reconstruction error of epoch
    def recon_error(y_true, y_pred):
        return K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true))
    
    h=X_train.shape[1]
    w=X_train.shape[2]
    colors=X_train.shape[3]
    
    # ENCODER
    input_vae = Input(shape=(h,w,colors))
    
    encoder_hidden1 = Conv2D(filters = b_f*4, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(input_vae)
    encoder_hidden1 = BatchNormalization()(encoder_hidden1)
    encoder_hidden1 = Activation('relu')(encoder_hidden1)
    
    encoder_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden1)
    encoder_hidden2 = BatchNormalization()(encoder_hidden2)
    encoder_hidden2 = Activation('relu')(encoder_hidden2)
    
    encoder_hidden3 = Conv2D(filters = b_f, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden2)
    encoder_hidden3 = BatchNormalization()(encoder_hidden3)
    encoder_hidden3 = Activation('relu')(encoder_hidden3)
    
    encoder_hidden4 = Flatten()(encoder_hidden3)
    
    # Latent Represenatation Distribution, P(z)
    z_mean = Dense(dim_representation, activation='linear', 
                            kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden4)
    z_std_sq_log = Dense(dim_representation, activation='linear', 
                            kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden4)
    
    # Sampling z from P(z)
    def sample_z(args):
        mu, std_sq_log = args
        epsilon = K.random_normal(shape=(K.shape(mu)[0], dim_representation), mean=0., stddev=1.)
        z = mu + epsilon * K.sqrt( K.exp(std_sq_log)) 
        return z
    
    z = Lambda(sample_z)([z_mean, z_std_sq_log]) #sample_z = a value. z_mean = layer, z_std_sq_log another layer. 
    
    
    # DECODER
    decoder_hidden0 = Dense(K.int_shape(encoder_hidden4)[1], activation='relu', kernel_initializer= initializers.he_normal(seed=None))(z)
    decoder_hidden0 = Reshape(K.int_shape(encoder_hidden3)[1:])(decoder_hidden0)
    
    decoder_hidden1 = Conv2DTranspose(filters = b_f, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden0)
    decoder_hidden1 = BatchNormalization()(decoder_hidden1)
    decoder_hidden1 = Activation('relu')(decoder_hidden1)
    
    decoder_hidden2 = Conv2DTranspose(filters = b_f*2, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden1)
    decoder_hidden2 = BatchNormalization()(decoder_hidden2)
    decoder_hidden2 = Activation('relu')(decoder_hidden2)
    
    decoder_hidden3 = Conv2DTranspose(filters = b_f*4, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden2)
    decoder_hidden3 = BatchNormalization()(decoder_hidden3)
    decoder_hidden3 = Activation('relu')(decoder_hidden3)
    
    decoder_hidden4a = Conv2D(filters = colors, kernel_size= 1, padding='valid', kernel_initializer = 'he_normal')(decoder_hidden3)
    model_final = Activation('sigmoid')(decoder_hidden4a)
    
    
    VAE = Model(input_vae, model_final)
    
    print(VAE.summary())
    print('### CNN-VAE INITIALIZED ###')
    print("### filters=%d : dim_representation=%d ###" % (b_f, dim_representation))     
    
    ##############################################################################################
    
    # Modelcheckpoints, optimizer
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    
    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='recon_error', patience=5),
                ModelCheckpoint(filepath=save_weight_name, monitor='recon_error', mode='min', 
                                save_weights_only=True, save_best_only=True)]

    # Optimizer for Training Neural Network
    optimizer_ = Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999) # Best learning rate = 0.00001
    VAE.compile(optimizer=optimizer_, loss = vae_loss, metrics = [recon_error])
    
    VAE.fit(X_train, X_train,
            shuffle=True,
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            validation_data=(X_test, X_test))
    VAE.save_weights(save_weight_name+'.h5')
    print('Model training Complete')
    print('Model saved to:', save_weight_name+'.h5', ". Use these weights for image reconstruction purposes.")
    

    return

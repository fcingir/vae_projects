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

# vae_loss function to perform stochastic gradient descent on
def vae_loss(y_true,y_pred):

    recontruction_loss = K.mean(K.binary_crossentropy(y_pred, y_true))
    latent_loss = -0.5 * K.mean(1 + z_std_sq_log - K.square(z_mean) - K.exp(z_std_sq_log), axis=-1 )
    return recontruction_loss + 0.01*latent_loss

# recon_error to measure reconstruction error of epoch
def recon_error(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true))
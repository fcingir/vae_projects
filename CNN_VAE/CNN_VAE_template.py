# Imports
import numpy as np

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Reshape, Dropout, Flatten, Activation, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import losses
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D 
from tensorflow.keras import regularizers

# Dimension of Latent Representation, filters, shape
dim_representation = 10
b_f = 64

# vae_loss function to perform stochastic gradient descent on
def vae_loss(y_true,y_pred):

    recontruction_loss = K.mean(K.binary_crossentropy(y_pred, y_true))
    latent_loss = -0.5 * K.mean(1 + z_std_sq_log - K.square(z_mean) - K.exp(z_std_sq_log), axis=-1 )
    return recontruction_loss + 0.01*latent_loss

# recon_error to measure reconstruction error of epoch
def recon_error(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true))

# ENCODER
input_vae = Input(shape=(32,32,3))

encoder_hidden1 = Conv2D(filters = b_f, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(input_vae)
encoder_hidden1 = BatchNormalization()(encoder_hidden1)
encoder_hidden1 = Activation('relu')(encoder_hidden1)

encoder_hidden2 = Conv2D(filters = b_f, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden1)
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

decoder_hidden2 = Conv2DTranspose(filters = b_f, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden1)
decoder_hidden2 = BatchNormalization()(decoder_hidden2)
decoder_hidden2 = Activation('relu')(decoder_hidden2)

decoder_hidden3 = Conv2DTranspose(filters = b_f, kernel_size = 2, padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden2)
decoder_hidden3 = BatchNormalization()(decoder_hidden3)
decoder_hidden3 = Activation('relu')(decoder_hidden3)

decoder_hidden4a = Conv2D(filters = 3, kernel_size= 1, padding='valid', kernel_initializer = 'he_normal')(decoder_hidden3)
model_final = Activation('sigmoid')(decoder_hidden4a)


VAE = Model(input_vae, model_final)

# Encoder Model
Encoder = Model(inputs = input_vae, outputs = [z_mean, z_std_sq_log])
no_of_encoder_layers = len(Encoder.layers)
no_of_vae_layers = len(VAE.layers)

# Decoder Model
decoder_input = Input(shape=(dim_representation,))
decoder_hidden = VAE.layers[no_of_encoder_layers+1](decoder_input)

for i in np.arange(no_of_encoder_layers+2 , no_of_vae_layers-1):
    decoder_hidden = VAE.layers[i](decoder_hidden)
decoder_hidden = VAE.layers[no_of_vae_layers-1](decoder_hidden)
Decoder = Model(decoder_input,decoder_hidden)

print(VAE.summary())
print('### CNN-VAE INITIALIZED ###')
print("### filters=%d : dim_representation=%d ###" % (b_f, dim_representation))     

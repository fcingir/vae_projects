# Data unpacking utilities
import numpy as np

def prepare_dataset_mat():

    import scipy.io as sio
    
    directory = 'C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\svhn'
    path_train = directory + '/train_32x32.mat'
    directory = 'C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\svhn'
    path_test = directory + '/test_32x32.mat'
    
    train_data = sio.loadmat(path_train)
    test_data = sio.loadmat(path_test)
    
    # access to the dict
    x_train = train_data['X']/255.0
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    y_train = train_data['y']
    x_test = test_data['X']/255.0
    x_test = np.transpose(x_test, (3, 0, 1, 2))
    y_test = test_data['y']
    
    return x_train, x_test, y_train, y_test
	
def prepare_dataset_fashion_mnist():
    import gzip
    
    directory = 'C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\fashionmnist'
    file = directory + '/t10k-images-idx3-ubyte.gz'
    filePath_test_set = file
    
    directory = 'C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\fashionmnist'
    file = directory + '/t10k-labels-idx1-ubyte.gz'
    filePath_test_label = file
    
    directory = 'C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\fashionmnist'
    file = directory + '/train-images-idx3-ubyte.gz'
    filePath_train_set = file
    
    directory = 'C:\\Users\\31687\\Desktop\\draw_tensorflow\\draw\\fashionmnist'
    file = directory + '/train-labels-idx1-ubyte.gz'
    filePath_train_label = file
    
    with gzip.open(filePath_test_label, 'rb') as trainLbpath:
         y_test = np.frombuffer(trainLbpath.read(), dtype=np.uint8,
                                   offset=8)
    
    with gzip.open(filePath_test_set, 'rb') as trainSetpath:
         X_test = np.frombuffer(trainSetpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(y_test), 28, 28, 1)/255.0
            
    with gzip.open(filePath_train_label, 'rb') as trainLbpath:
         y_train = np.frombuffer(trainLbpath.read(), dtype=np.uint8,
                                   offset=8)
    with gzip.open(filePath_train_set, 'rb') as trainSetpath:
         X_train = np.frombuffer(trainSetpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(y_train), 28, 28, 1)/255.0
        
    return X_train, X_test, y_train, y_test

def prepare_dataset_mnist():
	from keras.datasets import mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data() 
	
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
	
	return x_train, x_test, y_train, y_test
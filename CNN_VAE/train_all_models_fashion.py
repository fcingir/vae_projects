#Model training block
import numpy as np
import os
from train_CNN_VAE_create_folders import train_model
from dataset_unpacking_utility import prepare_dataset_mat, prepare_dataset_fashion_mnist, prepare_dataset_mnist

dataset_prep = np.asarray([prepare_dataset_mnist])
model_hyperparameters = [#[2,32],[10,32],[20,32],[100,32],
                         #[2,64],[10,64],[20,64],[100,64],
                         #[2,128],[10,128],[20,128],
                         [100,128]]

def train_all_datasets(add_list_of_datasets):
    #create names for weights_test
    weight_names_list = []
    for i in range(len(dataset_prep)):
        filename = 'dataset_'+str(i)+'_run_initial'
        weight_names_list.append(filename)
    weight_names_list = np.array(weight_names_list)
    
    return_here_path = os.getcwd()
    for i in range(len(dataset_prep)):
        os.chdir(return_here_path)
        
        X_train, X_test, y_train, y_test = dataset_prep[i]() #cycle through each function to create dataset
        weights_name_dataset = weight_names_list[i] #create basename for folder to be created for a particular dataset
        
        #automate folder creation for each dataset
        path_dataset = os.path.abspath(os.getcwd())+'\\'+str(weights_name_dataset)
        if not os.path.exists(path_dataset):
            os.makedirs(path_dataset)
        
        for j in model_hyperparameters:
            os.chdir(path_dataset) 
            weights_name = weights_name_dataset+'_ndim_'+str(j[0])+'_filters_'+str(j[1])
            train_model(X_train, X_test, epochs = 100, dim_representation = j[0], b_f = j[1], save_weight_name = weights_name) #(train_set, test_set, modelname, desired epochs, filename to save to)
            
    os.chdir(return_here_path) #return to working directory of this script
    return

train_all_datasets(dataset_prep)
    
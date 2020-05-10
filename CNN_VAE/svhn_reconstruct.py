
#Reconstruction block. FULLY FUNCTIONAL SINCE 19/02/2020. Memory problems after 2 folders though

import numpy as np
import os
from VAE_reconstruct import reconstruct_images
from dataset_unpacking_utility import prepare_dataset_mat

model_hyperparameters = [[2,32],[10,32],[20,32],
                         [2,64],[10,64],[20,64],
                         [2,128],[10,128],[20,128]
                        ]
dataset_prep = np.asarray([prepare_dataset_mat])

def reconstruct_all_datasets(add_list_of_datasets): # seeks folders in current working directory, enters them and performs reconstructions in the folder
    #grab npy 
    import os
    path = "C:\\Users\\31687\\Desktop\\VAE-SVHN-master\\Automated\\dataset_svhn_run_initial"
    _, X_test, _, _ = dataset_prep[0]() #cycle through each function to create dataset
    
    del _
    os.chdir(path)
    
    reconstruction_errors_filepaths = []
    for file in os.listdir():
        if file.endswith(".h5"):
            if 'mnist' in file:
                print(root)
                os.chdir(root)
                print(os.getcwd())
                _, X_test, _, _ = dataset_prep[0]() #cycle through each function to create dataset
                
                
                for j in model_hyperparameters:
                    if "ndim_"+str(j[0])+"_filters_"+str(j[1]) in file:
                        reconstruct_images(X_test, dim_representation=j[0], b_f=j[1], filename_weights= file[:-3])
                        print('Reconstructing model with hyperparams:' + str(j))
                    
            if 'fashion' in file:
                _, X_test, _, _ = dataset_prep[1]() #cycle through each function to create dataset
                
                os.chdir(root)
                for j in model_hyperparameters:
        
                    if "ndim_"+str(j[0])+"_filters_"+str(j[1]) in file:
                        reconstruct_images(X_test, dim_representation=j[0], b_f=j[1], filename_weights= file[:-3])
                        print('Reconstructing model with hyperparams:' + str(j))
            
            if 'svhn' in file:
                
                os.chdir(path)
                for j in model_hyperparameters:
                    if "ndim_"+str(j[0])+"_filters_"+str(j[1]) in file:
                        reconstruct_images(X_test, dim_representation=j[0], b_f=j[1], filename_weights= file[:-3])
                        print('Reconstructing model with hyperparams:' + str(j))               
            
    os.chdir(path) #return to working directory of this script
    return

reconstruct_all_datasets(dataset_prep)


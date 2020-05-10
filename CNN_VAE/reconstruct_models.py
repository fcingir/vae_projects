import numpy as np

def average_recon_error_label(reconstruction_errors, labels_dataset):
    reconstruction_errors_labels = np.stack([labels_dataset,reconstruction_errors],axis=-1)
    unique_elements = np.unique(reconstruction_errors_labels[:,0])
    
    idx = []
    for i in unique_elements:
        indexes = []
        for element in reconstruction_errors_labels:
            if element[0] == i:
                indexes.append(element[1])
        indexes = np.array(indexes)
        idx.append(indexes)
    idx = np.array(idx)
    
    average_error_label = np.around(np.array([np.mean(a) for a in idx]), decimals = 4)
    print(average_error_label)

    return average_error_label

# errors = average_recon_error_label(reconstruction_error, y_test)  #usage example


def gather_all_recon_errors():
    #grab npy 
    import os
    return_here_path = os.getcwd()
    
    reconstruction_errors_filepaths = []
    for root, dirs, files in os.walk(return_here_path):
        for file in files:
            if file.endswith("recon_error.npy"):
                filepath = os.path.join(root, file)
                #print(filepath)
                reconstruction_errors_filepaths.append(filepath)
    
    #reconstructed_images_filepaths = []
    #for root, dirs, files in os.walk(return_here_path):
    #    for file in files:
    #        if file.endswith("reconstructed_images.npy"):
    #            filepath = os.path.join(root, file)
    #            #print(filepath)
    #            reconstructed_images_filepaths.append(filepath)
                  
    return np.array(reconstruction_errors_filepaths)

filepaths_recon_errors = gather_all_recon_errors() #the next function depends on this variable. integrate into function.

def model_selection_lowest_recon_error(filepaths_recon_errors):
    from dataset_unpacking_utility import prepare_dataset_mat, prepare_dataset_fashion_mnist, prepare_dataset_mnist
    _, _, _, y_test = prepare_dataset_mnist()
    
    avg_recon_error_per_model = []
    model_name = []
    paths = []
    for i in filepaths_recon_errors:
        print('File: '+i+' label averages equal:')
        model_averages = average_recon_error_label(np.load(i), y_test)
        model_averages
        global_average = sum(model_averages)/len(model_averages)
        
        
        paths.append(i)
        avg_recon_error_per_model.append(global_average) #calculates total average
        model_name.append(i.rsplit('dataset_', 1)[1]) #save model name concisely
        
        #print(global_average, 'global average of model')
        print('\n')
            
    
    model_and_error = np.stack((np.array(avg_recon_error_per_model), np.array(model_name), np.array(paths)),axis=0)
    return model_and_error


df = model_selection_lowest_recon_error(filepaths_recon_errors).T
errors = df[0]
names = df[1]
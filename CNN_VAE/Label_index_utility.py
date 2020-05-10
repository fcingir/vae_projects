# Labeling utility
import numpy as np

def label_indexer(y_data):
    idx = []
    labelnumbers = np.arange(10)
    
    for labelnumber in labelnumbers:
        index_numbers = []
        for idy in range(len(y_data)):
            if y_data[idy] == labelnumber:
                index_numbers.append(idy)
        idx.append(np.array(index_numbers))
    idx = np.array(idx)
    idx[0]
    
    return idx
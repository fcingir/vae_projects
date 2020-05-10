#Visualize X_train, X_test or reconstructed images. Necessary format = (N,width,height)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Label_index_utility import label_indexer

def visualize_images(img_data, label_data, label_number):
  
    sorted_data=label_indexer(label_data)
    
    w = img_data.shape[1]
    h = img_data.shape[2]
    fig = plt.figure(figsize=(8, 9)) #basic setting = (9,13)
    columns = 5 #basic setting = 4
    rows = 5 #basic setting = 
    
    ## prep (x,y) for extra plotting
    #xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    #ys = np.abs(np.sin(xs))           # absolute of sine
    
    # ax enables access to manipulate each of subplots
    ax = []
    
    def grid():
        for i in range(columns*rows):
            img = img_data[sorted_data[label_number][i]].reshape(28,28)
            # create subplot and append to ax
            ax.append( fig.add_subplot(rows, columns, i+1) )
            
            plt.tick_params(
            axis='both',        # changes apply to both axes. Can set to 'y' or 'x'.
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=False, # labels along the bottom edge are off
            labelleft=False)
            
            
            ax[-1].set_title("id:"+str(sorted_data[label_number][i]))  # set title
            plt.imshow(img, alpha=1, cmap = matplotlib.cm.binary, interpolation="nearest")
        ## do extra plots on selected axes/subplots
        ## note: index starts with 0
        #ax[2].plot(xs, 3*ys)
        #ax[19].plot(ys**2, xs)
        
        plt.show()  # finally, render the plot
    grid()
    
    return


#compares two images 
def visualize_images_comparison(img_data, reconstructed_data, label_data, label_number):
  
    sorted_data=label_indexer(label_data)
    
    w = img_data.shape[1]
    h = img_data.shape[2]
    fig = plt.figure(figsize=(2, 15)) #basic setting = (9,13)
    columns = 2 #basic setting = 4
    rows = 10 #basic setting = 
    
    ## prep (x,y) for extra plotting
    #xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    #ys = np.abs(np.sin(xs))           # absolute of sine
    
    # ax enables access to manipulate each of subplots
    ax = []
    
    def grid():
        for i in range(columns*rows):
            if i % 2 == 0:
                img = img_data[sorted_data[label_number][i]].reshape(28,28)
                # create subplot and append to ax
                ax.append( fig.add_subplot(rows, columns, i+1) )
                
                plt.tick_params(
                axis='both',        # changes apply to both axes. Can set to 'y' or 'x'.
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False)
                
                
                ax[-1].set_title("img_id:"+str(sorted_data[label_number][i]))  # set title
                plt.imshow(img, alpha=1, cmap = matplotlib.cm.binary, interpolation="nearest")
                
            else:
                img = reconstructed_data[sorted_data[label_number][i-1]].reshape(28,28)
                # create subplot and append to ax
                ax.append( fig.add_subplot(rows, columns, i+1) )
                
                plt.tick_params(
                axis='both',        # changes apply to both axes. Can set to 'y' or 'x'.
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False)
                
                
                ax[-1].set_title("id:"+str(sorted_data[label_number][i-1]))  # set title
                plt.imshow(img, alpha=1, cmap = matplotlib.cm.binary, interpolation="nearest")
        ## do extra plots on selected axes/subplots
        ## note: index starts with 0
        #ax[2].plot(xs, 3*ys)
        #ax[19].plot(ys**2, xs)
        
        plt.show()  # finally, render the plot
    grid()
    
    return
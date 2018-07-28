import pandas as pd
import numpy as np
import os

from utils.general_utils import get_imgMatrix_from_id, get_id_from_filename, populate_breeds, get_label_array_from_id


#convert a directory of pre-processed images into a pandas dataframe for easy use with tensorflow
def dataFrameBuilder(data_amount=100,
                     start_index=0,
                     ret_input=True,
                     ret_output=False,
                     dir="../data/preprocessed_data/Train/"):
    #df = pd.DataFrame(columns=['ID', 'Image Data', 'Breed'])
    d = []

    BREED_LIST = "../data/preprocessed_data/breed_list.csv"  
    
    #prepare the breed list dataframe
    labels = populate_breeds(BREED_LIST) #get the list of all dog breeds
    labels_np = np.array(labels).reshape(120,1) #labels list reshaped to numpy array
            
    data_files = os.listdir(dir) #get a list of all filenames from Train dir

    f = 0        #current file
    counter = 0  #loop counter
    
    #while != last file
    for file in data_files:
        #print("File: " , f)
        
        file_id = get_id_from_filename(file)
        data = get_imgMatrix_from_id(file_id)
        breed_matrix = get_label_array_from_id(file_id, labels_np)
        if(ret_input):
            d.append(data)
            counter += 1
        if(ret_output):
            d.append(breed_matrix)
            counter+=1
            
        if(counter >= data_amount): #if data loaded into ram == data_amount, return loaded data as a dataframe
            #df = pd.DataFrame(d, columns=['ID', 'Image Data', 'Breed']) #store list in a temp dataframe            
            return d
            break
            
        f +=1

           
    return d
import pandas as pd
import numpy as np
import os

from utils.general_utils import get_imgMatrix_from_id, get_id_from_filename, populate_breeds, get_label_array_from_id


#convert a directory of pre-processed images into a pandas dataframe for easy use with tensorflow
def dataFrameBuilder(data_amount=None,
                     start_index=None,
                     dir="../data/preprocessed_data/Train/"):
    df = pd.DataFrame(columns=['ID', 'Image Data', 'Breed'])
    d = []

    BREED_LIST = "../data/preprocessed_data/breed_list.csv"  
    
    #prepare the breed list dataframe
    labels = populate_breeds(BREED_LIST) #get the list of all dog breeds
    labels_np = np.array(labels).reshape(120,1) #labels list reshaped to numpy array
            
    data_files = os.listdir(dir) #get a list of all filenames from Train dir

    f = 0
    counter = 0
    
    for file in data_files:
        f +=1
        print("File: " , f)
        
        file_id = get_id_from_filename(file)
        data = get_imgMatrix_from_id(file_id)
        breed_matrix = get_label_array_from_id(file_id, labels_np)
        
        d.append({'ID': file_id, 'Image Data': data, 'Breed': breed_matrix})
        
        if(counter > 10): #every 10 indexes in order to preserve ram
            df_temp = pd.DataFrame(d, columns=['ID', 'Image Data', 'Breed']) #store list in a temp dataframe            
            
            df = pd.concat([df, df_temp]) #concatenate the temp df onto the end of df
            
            d = [] #clear the list 
            counter = 0 #restart the counter
            
        
        if(data_amount != None and data_amount >= f):
            df_temp = pd.DataFrame(d, columns=['ID', 'Image Data', 'Breed']) #store list in a temp dataframe            
            
            df = pd.concat([df, df_temp]) #concatenate the temp df onto the end of df
            
            d = [] #clear the list             
            
            break
    
    df_temp = pd.DataFrame(d) #initialize the DataFrame
    df = pd.concat([df, df_temp])
    df_temp = None
    
    return df

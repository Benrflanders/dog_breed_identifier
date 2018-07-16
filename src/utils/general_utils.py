#genearl utility functions used throughout the project

import os

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2 #used for image to matrix conversion

def get_imgMatrix_from_id(image_id, image_dir="../data/preprocessed_data/Train", filetype=".png"):
    image_loc = image_dir + "/" + image_id + "" + filetype

    #return plt.imread(image_loc)
    #image = tf.image.decode_jpeg(image_loc) #Decode a JPEG-encoded image to a uint8 tensor
    #resized_img = tf.image.resize_images(image,(32,32)) #resize all images to 250x250
    
    image = cv2.imread(image_loc)
    return image
    
def get_filename_from_id(image_id, image_dir="../data/preprocessed_data/Train", filetype=".png"):
    return image_dir + "/" + image_id + "" + filetype

#return the breed associated with an id
def get_breed_from_id(id, filename="../data/included/labels.csv"):
    
    #access training data labels from labels.csv
    training_data = pd.read_csv(filename)
    
    #get the one row where the id is the id supplied
    #training_data[training_data['id'].str.match(id)]
    id_row = training_data[training_data['id'].str.match(id)]
    
    #get the breed from that row
    breed_name = id_row['breed'].values
    
    return breed_name

def get_breed_value_from_id(id, labels_list, filename="../data/included/labels.csv"):
    
    breed_name = get_breed_from_id(id, filename) #get the breed name of the current id
    #print("breed_name: " + breed_name) #testing... delete this line 
    
    target_array = [0.0] * len(labels_list)        
    
    target_value = 0
    
    #features/labels are coded from [0,len(labels_list)]
    
    for i in range(len(labels_list)):
        if(labels_list[i] == breed_name): #each sample will only have one instance where this is true
            target_array[i] = 1.0
            
            #need a better option than just multiplying by i... look into using prime numbers
            target_value = i * 1.0
            
            
            
    '''
    
    it = np.nditer(labels_list)
    while not it.finished:
        #print("index is %s... %s" % (it.index, it[0]))
        if(it[0] == breed_name):
            target_array[it.index] = 1.0
        
        it.iternext()
        
    '''
    '''
    #target_array = np.zeros(len(labels_list))

    for index, iterindex in np.nditer(labels_list): #index contains the name of the dog breed, 
        print(index)
        if(index == breed_name):
            print("TRUE!")
            print(iterindex)
            target_array[iterindex] = 1.0
        #if(labels_list[index] == breed_name):
            #target_array[index] = 1.0
    '''          
    #return a single integer
    return target_value




#get the number of files in a directory
def count_files(dir="data/Train"):

    data_files = os.listdir(dir)    
    #filename = data_files[random_num]
    #id = filename.rsplit(".",1)[0]
    
    
    counter = 0
    for files in data_files:
        counter += 1
        #print(counter, files)
    
    return counter

def get_id_from_filename(file):
    return file.rsplit(".",1)[0]

def get_random_id(dir="../data/included/Train"):
    data_files = os.listdir(dir)    
    random_num = np.random.randint(0,high=len(data_files))
    filename = data_files[random_num]
    random_id = filename.rsplit(".",1)[0]
                                   
    return random_id


def populate_breeds(breedList):
    breed_labels = [] #breed label list is already sorted
    
    raw_labels = pd.read_csv(breedList)
        
    for columns in raw_labels:
        breed_labels.append(columns)
            
    return breed_labels


def get_num_training_files_per_breed(breed_list):
    breed_occur = { }
    
    for breed in breed_list:
        breed_occur[breed] = 1
        for file in data_files:
            curr_id = get_id_from_filename(file)
            curr_breed = get_breed_from_id(curr_id)
            if curr_breed == breed:
                breed_occur[breed] += 1
            
    return breed_occur


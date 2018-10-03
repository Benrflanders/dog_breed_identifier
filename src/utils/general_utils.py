#genearl utility functions used throughout the project

import os

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2 #used for image to matrix conversion
from scipy.io import loadmat


def get_imgMatrix_from_id(image_id, image_dir="../data/preprocessed_data/Train", filetype=".png"):
    image_loc = image_dir + "/" + image_id + "" + filetype

    #return plt.imread(image_loc)
    #image = tf.image.decode_jpeg(image_loc) #Decode a JPEG-encoded image to a uint8 tensor
    #resized_img = tf.image.resize_images(image,(32,32)) #resize all images to 250x250
    
    image = cv2.imread(image_loc)
    image = np.array(image)
    image = np.expand_dims(image, axis = 0) #add a dimension for keeping track of the batch index
    return image
    
def get_filename_from_id(image_id, image_dir="../data/preprocessed_data/Train", filetype=".png"):
    return image_dir + "/" + image_id + "" + filetype

def get_breed_from_id(id, file_list):
    mat = loadmat(file_list)
    for file in mat.get('file_list'):
        curr_id = (file[0][0]).split('/')
        curr_id = get_id_from_filename(curr_id[1])
        
        if(id == curr_id):
            breed = (file[0][0]).split('/')[0]
            breed = ''.join(breed.split('-')[1:])
            return breed.lower()


#generate a list that contains the id of a breed paired with its actual breed name
def generate_breed_list(dir_name = '../data/raw/Images/'):
    folders = os.listdir(dir_name)
    index = 0

    breed_dict = []
    
    while index < len(folders):
        if(folders[index][0] == '.'):
            folders[index].pop() #if it is a hidden file, don't include it

        #breed_id, breed_name = folders[index].split('-')
        split = folders[index].split('-')

        breed_id = split[0]
        breed_name = ''.join(split[1:])
        
        print("id: ", breed_id)
        #breed_dict.append("breed_id" = breed_id, "breed_name" = breed_name)
        print("name: ", breed_name)
        breed_list.append([breed_id, breed_name])
        index+= 1
        
        
        
        
#commonly used*********** careful changing this
def get_breed_value_from_id(id, labels_list, file_list):
    
    breed_name = get_breed_from_id(id, file_list) #get the breed name of the current id
    #print("breed_name: " + breed_name) #testing... delete this line 
    
    target_array = [0.0] * len(labels_list)        
    
    target_value = 0
    #features/labels are coded from [0,len(labels_list)]
    
    for i in range(len(labels_list)):
        if(labels_list[i] == breed_name): #each sample will only have one instance where this is true
            target_array[i] = 1.0
            target_array = np.array(target_array)
            target_array = np.expand_dims(target_array, axis = 0) #add a dimension for keeping track of the batch index

            
            #return np.reshape(target_array, [120,1])
            return target_array
    print("breed name: ", breed_name)
    #return a single integer
    return target_array





#takes in an id, retrieves the breed of that id, then returns an array 
#with a single 1 in the corresponding index of that breed
def get_label_array_from_id(id, labels_list, filename="../data/included/labels.csv"):
    breed_name = get_breed_from_id(id, filename) #get the breed name of the current id
    
    label_array = [] #the array that will be returned
    
    for i in range(len(labels_list)):
        if(labels_list[i] == breed_name): #each sample will only have one instance where this is true
            label_array.append(1)
        else:
            label_array.append(0)
    return label_array
    
    
    
    
    
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

def get_id_from_filename_old(file):
    return file.rsplit(".",1)[0]


def get_id_from_filename(file):
    return file.split('.')[0]
    
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




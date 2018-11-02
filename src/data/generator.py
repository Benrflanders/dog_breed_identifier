import os
import math
import numpy as np
import pandas as pd
from random import shuffle

from utils.general_utils import *

#generates data for direct use with the main model builder script

class generator():
    def __init__(self, batch_size=1, number_of_images=20580):
        self.batch_size = batch_size #optional

        self.BREED_LIST = 'H:/Dog_Breed_project/processed/breed_list.csv'
        #self.train_dir = '../data/processed/train_images/'
        self.train_dir = 'H:/Dog_Breed_project/processed/train_images/'
        self.test_dir = 'H:/Dog_Breed_project/processed/test_images/'
        self.train_list = 'H:/Dog_Breed_project/train_list.mat'
        self.test_list = 'H:/Dog_Breed_project/test_list.mat'

        #prepare the breed list array
        self.labels = populate_breeds(self.BREED_LIST) #get the list of all dog breeds
        self.labels = np.array(self.labels).reshape(120,1) #labels list reshaped to numpy array

        self.start = 0
        self.curr = 0
        self.end = number_of_images

        self.batch_features = np.zeros((self.batch_size, 500, 500, 3))
        self.batch_labels = np.zeros((batch_size,120))

    #yield a batch of training data
    def generate_training_data_batch(self):
        #get from train_list.mat
        #os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
        file_list = os.listdir('H:/Dog_Breed_project/processed/train_images/')
        shuffle(file_list)
    
        X = []
        y = []
        
        counter = 0 #number of files add so far
        index = 0 #index in the current file list
        while True:
            if(counter == 0):
                X = []
                y = []
            
            #the end of the current list order is reached
            if(index == len(file_list)): 
                shuffle(file_list)
                index = 0
                #so shuffle again and start from the beginning of the list

            curr_file = file_list[index]
            curr_file_id = get_id_from_filename(curr_file)
            
            curr_file_matrix = get_imgMatrix_from_id(curr_file_id, image_dir=self.train_dir)
            
            curr_file_label = get_breed_value_from_id(curr_file_id, self.labels, self.train_list)

            X.append(curr_file_matrix)
            y.append(curr_file_label)
            counter += 1 
            index += 1
            if((counter % self.batch_size) == 0):
                counter = 0
                #X = np.array(X)
                #X = np.reshape(X, [self.batch_size, 500,500, 3])
                X = np.vstack(X)
                y = np.vstack(y)
                yield X, y
                X = []
                y = []

                
    #yield a batch of training data
    def generate_testing_data_batch(self):
        #get from train_list.mat
        #os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
        file_list = os.listdir('H:/Dog_Breed_project/processed/test_images/')
        self.shuffle_data(file_list)
    
        X = []
        y = []
        
        counter = 0 #number of files add so far
        index = 0 #index in the current file list
        while True:
            
            #the end of the current list order is reached
            if(index == len(file_list)): 
                shuffle(file_list)
                index = 0
                #so shuffle again and start from the beginning of the list

            curr_file = file_list[index]
            curr_file_id = get_id_from_filename(curr_file)
            
            curr_file_matrix = get_imgMatrix_from_id(curr_file_id, image_dir=self.test_dir)
            
            curr_file_label = get_breed_value_from_id(curr_file_id, self.labels, self.test_list)

            X.append(curr_file_matrix)
            y.append(curr_file_label)
            counter += 1 
            index += 1
            if((counter % self.batch_size) == 0):
                X = np.vstack(X)
                y = np.vstack(y)
                yield X, y
                X = []
                y = []
    
    
    #yield a single index of the training data
    def generate_training_data(self):
        #get from train_list.mat
        #os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
        file_list = os.listdir('../data/processed/train_images/')
        shuffle(file_list)

        index = 0 #index in the current file list
        while True:

            #the end of the current list order is reached
            if(index == len(file_list)): 
                shuffle(file_list)
                index = 0
                #so shuffle again and start from the beginning of the new list

            curr_file = file_list[index]
            curr_file_id = get_id_from_filename(curr_file)
            
            curr_file_matrix = get_imgMatrix_from_id(curr_file_id, image_dir=self.train_dir)   
            curr_file_label = get_breed_value_from_id(curr_file_id, self.labels, self.train_list)

            #curr_file_matrix = np.asarray(curr_file_matrix[0])
            #curr_file_label = np.asarray(curr_file_label[0])
            
            index += 1

            yield (curr_file_matrix, curr_file_label)

    #yield a single index of testing data
    def generate_testing_data(self):
        #get from train_list.mat
        #os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
        file_list = os.listdir('../data/processed/test_images/')
        self.shuffle_data(file_list)
    
        index = 0 #index in the current file list
        while True:
            
            #the end of the current list order is reached
            if(index == len(file_list)): 
                shuffle(file_list)
                index = 0
                #so shuffle again and start from the beginning of the list

            curr_file = file_list[index]
            curr_file_id = get_id_from_filename(curr_file)
            
            curr_file_matrix = get_imgMatrix_from_id(curr_file_id, image_dir=self.test_dir)
            curr_file_label = get_breed_value_from_id(curr_file_id, self.labels, self.test_list)

            index += 1
            
            yield curr_file_matrix, curr_file_label
                
    def shuffle_data(self, file_list):
        
        return shuffle(file_list)

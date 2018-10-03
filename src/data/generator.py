import os
import math
import numpy as np
import pandas as pd
from random import shuffle

from utils.general_utils import *

#generates data for direct use with the main model builder script

class generator():
    def __init__(self, batch_size, features, number_of_images=20580):
        self.batch_size = batch_size

        self.BREED_LIST = '../data/processed/breed_list.csv'
        self.train_dir = '../data/processed/train_images/'
        self.test_dir = '../data/processed/test_images/'
        self.train_list = '../data/raw/train_list.mat'
        self.test_list = '../data/raw/test_list.mat'

        #prepare the breed list array
        self.labels = populate_breeds(self.BREED_LIST) #get the list of all dog breeds
        self.labels = np.array(self.labels).reshape(120,1) #labels list reshaped to numpy array

        self.start = 0
        self.curr = 0
        self.end = number_of_images

        self.batch_features = np.zeros((self.batch_size, 500, 500, 3))
        self.batch_labels = np.zeros((batch_size,120))

    def next(self):
        self.curr+=1
        self.get_data()

    #yield a batch of training data
    def generate_training_data(self):
        #get from train_list.mat
        os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
        file_list = os.listdir('../data/processed/train_images/')
        shuffle(file_list)
    
        X = []
        y = []
        
        counter = 0
        while True:
            curr_file = file_list[counter]
            curr_file_id = get_id_from_filename(curr_file)
            
            curr_file_matrix = get_imgMatrix_from_id(curr_file_id, image_dir=self.train_dir)
            
            curr_file_label = get_breed_value_from_id(curr_file_id, self.labels, self.train_list)

            X.append(curr_file_matrix)
            y.append(curr_file_label)
            counter += 1
            if((counter % self.batch_size) == 0):
                yield X, y
                X = []
                y = []
            if(counter == len(file_list)-1):
                X = []
                y = []
                break


    def get_testing_data(self):
        #get from test_list.mat

        return True

    def shuffle_data(self):
        
        return True

    def data_generator(self):
        while True:
            for i in range(self.batch_size):     
                # choose random index in features
                index= random.choice([len(features),1])
                X[i] = train_input_fn(index=index)
                y[i] = 1#get the label of the current input data
            yield X, y

    def get_input_data(self, index=0, data_amnt=1):
        input_img_data = np.asarray(input_img_data)
        return input_img_data

    def get_test_input_data(self):

        return input_img_data
    
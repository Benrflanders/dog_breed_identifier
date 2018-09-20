import os
import os
import numpy as np #linear algebra
import pandas as pd #data preprocessing
import matplotlib.pyplot as plt #data visualization
import h5py
import PIL

import data.generator as generator

class inception_classifier():
    def __init__(self):
        #basic parameters
        self.image_size = 500
        self.batch_size = 1
        self.num_classes = 120
        self.train = True

        #get inception_v3 neural network classifier
        self.inception_v3 = self.get_inception_v3()
        
        #build model

        #access generator
        gen = generator()

        
    def input_layer(self):
        x = tf.keras.layers.Input(shape=(500,500,3), batch_size=batch_size,name='input_data',dtype='float32')
        
        return x

    def output_layer(self):
        y = tf.placeholder(tf.float32, shape=[None, 120,1], name='correct_labels')
        return y

    def predicted_layer(self):
        y_pred = tf.placeholder(tf.float32, shape=[None,120,1], name='predicted_labels')
        return y_pred

    def train_input_fn(index=0, data_amnt=1):
        input_img_data = dataFrameBuilder(data_amount=data_amnt,
                                          start_index=index)

        input_img_data = np.asarray(input_img_data)

        return input_img_data
    
    def get_inception_v3(self):
        inception_v3 = tf.keras.applications.InceptionV3(include_top=False,
                                                input_tensor=x,
                                                classes=120)

        return inception_v3

    def train_model(self):
        self.model = tf.keras.Model(inputs=inception_v3.input, output=self.predictions)

    def save_model(self):
        
    def load_model(self):
    

import os
import numpy as np #linear algebra
import pandas as pd #data preprocessing
#import matplotlib.pyplot as plt #data visualization
import h5py
import PIL

import tensorflow as tf

import data.generator as generator #geneartes a single batch of data

class inception_classifier():
    def __init__(self):
        #basic parameters
        self.image_size = 500
        self.batch_size = 1
        self.num_classes = 120
        self.train = True
        
        #set the input and output placeholders
        #self.X = tf.placeholder(tf.float32, shape=([None, 500,500,3]))
        #self.y = tf.placeholder(tf.float32, shape=([None, 120, 1]))

        self.X = tf.keras.layers.Input(shape=(500,500,3), batch_size=self.batch_size,name='input_data',dtype='float32')
        self.y = tf.placeholder(tf.float32, shape=[None, 120,1], name='correct_labels')
        self.y_pred = tf.placeholder(tf.float32, shape=[None,120,1], name='predicted_labels')
        
        #get inception_v3 neural network classifier
        self.model = self.get_inception_v3()

        #update the input and output layers of the model
        #self.update_output_layer()
        self.model = tf.keras.Model(inputs=self.model.input, outputs=self.generate_output_layer())

        #access the generator for use during any training/testing sess
        self.gen = generator.generator(self.batch_size)
        
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)
            print(i, layer.input)
            print(i, layer.output)


        #build model
        self.train_model()

        #evaulate the model
        
        
    def input_layer(self):
        x = tf.keras.layers.Input(shape=(500,500,3), batch_size=self.batch_size,name='input_data',dtype='float32')
        
        return x

    def output_layer(self):
        y = tf.placeholder(tf.float32, shape=[None, 120,1], name='correct_labels')
        return y

    def prediction_layer(self):
        y_pred = tf.placeholder(tf.float32, shape=[None,120,1], name='predicted_labels')
        return y_pred
    
    def get_inception_v3(self):
        inception_v3 = tf.keras.applications.InceptionV3(include_top=False,
                                                #weights='imagenet',
                                                input_tensor=self.X,
                                                classes=120)

        return inception_v3

    def train_op(self):
        return None

    def generate_output_layer(self):
        #steps for adding a new output layer
        output_layer = self.model.output
        output_layer = tf.keras.layers.GlobalAveragePooling2D()(output_layer) #replace the current global avg pool 2d
        output_layer = tf.keras.layers.Dense(1024, activation='relu')(output_layer) 
        predictions = tf.keras.layers.Dense(120, activation='softmax')(output_layer) #120 classes in the new model
        #self.model = tf.keras.Model(inputs=self.model.input, outputs=predictions)
        return predictions
    
    def train_model(self):

        #self.model = tf.keras.Model(inputs=self.model.input, output=self.predictions)

        #training configurations that worked on my desktop -- see specs report
        #config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.allocator_type = 'BFC'
        #config.gpu_options.per_process_gpu_memory_fraction = 0.40
        #config.gpu_options.allow_growth = True

        #tf.reset_default_graph()
        print("about to start training :D")
        
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
            sess.run(tf.global_variables_initializer())

            #initialize the main layers
            #x = self.input_layer()
            #y = self.output_layer()
            #y_pred = self.prediction_layer()
            
            #train using the train op
            # for i in range(1000):
            #    sess.run(train_op)
              
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='sgd')
            self.model.fit_generator(self.gen.generate_training_data(), steps_per_epoch=10, epochs=50)
              
            writer.close()
                
        #img_data = train_input_fn(index=index, data_amnt=batch_size)
        #breed_data = train_output_fn(index=index, data_amnt=batch_size)

       

    def evaluate_model(self):
        return self.model.evaluate_generator(self.gen.generate_testing_data(), steps=10, )
    
    def save_model(self):
        self.model.save('../../models/model.h5')
        return True
        
    def load_model(self):
        self.model = tf.keras.models.load_model('../..models/model.h5')
        print(self.model.summary())
        return True
    

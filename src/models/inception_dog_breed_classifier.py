import os
import numpy as np #linear algebra
import pandas as pd #data preprocessing
#import matplotlib.pyplot as plt #data visualization
import h5py
import PIL

import tensorflow as tf

import data.generator as generator #geneartes a single batch of data
import utils.general_utils as utils

class inception_classifier():
    def __init__(self):

        self.train = True
        self.test = True
        self.load_model = True
        
        #basic parameters - change these to fit the training data
        self.image_size = 224
        self.batch_size = 128
        self.num_classes = 120
        self.i = 0
        #self.is_training = tf.placeholder(tf.bool)

        self.EPOCHS = 25
        self.INITIALIZATION_EPOCHS = 3
        
        #set the input and output placeholders
        #self.X = tf.placeholder(tf.float32, shape=([None, 500,500,3]))
        #self.y = tf.placeholder(tf.float32, shape=([None, 120, 1]))

        
        self.X = tf.keras.layers.Input(shape=(self.image_size,self.image_size,3), batch_size=self.batch_size,name='input_data',dtype='float32')
        self.y = tf.placeholder(tf.float32, shape=[None, 120,1], name='correct_labels')
        self.y_pred = tf.placeholder(tf.float32, shape=[None,120,1], name='predicted_labels')
        
        #get pretrained neural network
        self.base_model = self.get_inception_resnet_v2()

        #update the input and output layers of the model
        self.model = tf.keras.Model(inputs=self.base_model.input, outputs=self.generate_output_layer())
        
        #access the generator for use during any training/testing sess
        self.gen = generator.generator(self.batch_size)

        if(self.i == 0 and self.train == True): #if on the first iteration of training
            print("Initial layers of inception resnet v2 will now be trained")
            self.initial_train() #train the last nodes on the new output classes                                                                     
                                                                        
        if(self.train):
            print("Program is now in training mode")
            self.train_model() #train the entire model

        #evaulate the model
        if(self.test):
            self.evaluate_model()

        quit()
        
            
    def get_inception_v3(self):
        inception_v3 = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet',
                                                input_tensor=self.X,
                                                classes=120)

        return inception_v3

    def get_inception_resnet_v2(self):
        inception_resnet_v2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                                                         weights='imagenet',
                                                                                         input_tensor=self.X,
                                                                                         classes=120)
        return inception_resnet_v2
                                                                                         
    
    def generate_output_layer(self):
        #steps for adding a new output layer
        output_layer = self.base_model.output
        output_layer = tf.keras.layers.GlobalAveragePooling2D()(output_layer) #replace the current global avg pool 2d
        output_layer = tf.keras.layers.Dense(1024, activation='relu')(output_layer) 
        predictions = tf.keras.layers.Dense(120, activation='softmax')(output_layer) #120 classes in the new model
        #self.model = tf.keras.Model(inputs=self.model.input, outputs=predictions)
        return predictions

    def initial_train(self):
        #freeze all inception_resent layers so that only the newly (randomly) generated layers can be trained
        for layer in self.base_model.layers:
            layer.trainable = False
    
        print("running initial train")

        for layer in self.model.layers[:780]:
            layer.trainable = False
        for layer in self.model.layers[780:]: #let's try only training the newly created final layers and leave the rest of
            #inception resnet untouched
            layer.trainable = True

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='rmsprop')
        self.model.fit_generator(self.gen.generate_training_data_batch(), steps_per_epoch=94, epochs=3) #up steps per epoch to 3k and epochs to ~4                            

        #save the model
        file_name = '../models/model_ckpt_' + str(self.i) + '.h5'
        self.model.save(file_name)
        
        #evaluate the newly saved model
        self.evaluate_model()

        return True
              
    
    def train_model(self):

        print("about to start training")
        
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("../Reports/log/", sess.graph)
            sess.run(tf.global_variables_initializer())

            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = False)
  
            #from tensorflow.keras.optimizers import SGD
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), options=run_opts)
            
            #try reducing learning rate to .0001
            

            #save checkpoints of the model during training...
            #self.i = 0
            if(self.i > 0):
                del self.model
                file_name = '../models/model_ckpt_' + str(self.i) + '.h5'
                self.model = tf.keras.models.load_model(file_name, compile=False)
                self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), options=run_opts)

            for layer in self.model.layers[:780]:
                layer.trainable = True
            for layer in self.model.layers[780:]: #let's try only training the newly created final layers and leave the rest of
                #inception resnet untouched
                layer.trainable = True
                
            while(self.i<=1): #each 'i' will be 1 epoch, 100 iterations in total
                
                #print("Running training for iterations: ", i*10, " to ", i*10+9)
                print("Running training for epoch: ", self.i)
                self.model.fit_generator(self.gen.generate_training_data_batch(), steps_per_epoch=94, epochs=1)
                #self.model.fit_generator(self.gen.generate_training_data(), steps_per_epoch=2, epochs=1)
                self.i+=1
                file_name = '../models/model_ckpt_' + str(self.i) + '.h5'
                self.model.save(file_name)

            self.model.save('../models/model.h5')
            writer.close()       

    def evaluate_model(self):

        print("about to start testing :D")
        del self.model
        self.model = tf.keras.models.load_model('../models/model_ckpt_' +  str(self.i) + '.h5')

        total_correct_predictions = 0
        i = 0
        while(i< 1):
            X,y = next(self.gen.generate_training_data_batch())
            classification = self.model.predict(X, batch_size=self.batch_size) #get a batch of predictions

            #check the accuracy of each member of the batch
            j = 0
            while(j < 128):
                pred_classification_top, pred_classification_second, pred_classification_third = utils.get_breed_from_output(classification[j])
                actual_classification, temp, temp2 = utils.get_breed_from_output(y[j]) #this function gives 3 outputs, but in the actual breed output only the first matters
                #check if the prediction was correct
                if(pred_classification_top == actual_classification):
                    total_correct_predictions += 1
                elif(pred_classification_second == actual_classification):
                    total_correct_predictions += 0
                elif(pred_classification_third == actual_classification):
                    total_correct_predictions += 0              
                j+=1
            i+=1 #run another batch                                                                           

        print("Accuracy = ", total_correct_predictions/(128*67))                     
        return True

    def use_model(self, X=None):
        print("test text to see if file is being updated at save")
        if(self.load_model):
            del self.model
            self.model = tf.keras.models.load_model('../models/model_ckpt_' + str(self.i) + '.h5')

        #test on training data to check if neural net is overfit
        X,y = next(self.gen.generate_training_data())
        
        classification = self.model.predict(X, batch_size=self.batch_size)
        print("classifications: ",classification)
        print("Predicted Breed for part 1: ", utils.get_breed_from_output(classification[0]))
        
        print("Actual Classification part 1: ", y[0])
        print("Actual Breed: ", utils.get_breed_from_output(y[0]))

        
        print("\nActual Classficiation part 2: ", y[1])

    
    def save_model(self):
        self.model.save('../models/model.h5')
        return True
        
    def load_model(self):
        self.model = tf.keras.models.load_model('../models/model.h5')
        print(self.model.summary())
        return True
    

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

        self.train = False
        self.test = False
        self.load_model = True
        #basic parameters
        self.image_size = 500
        self.batch_size = 4
        self.num_classes = 120
        self.train = True
        self.i = 0
        #self.is_training = tf.placeholder(tf.bool)
        
        #set the input and output placeholders
        #self.X = tf.placeholder(tf.float32, shape=([None, 500,500,3]))
        #self.y = tf.placeholder(tf.float32, shape=([None, 120, 1]))

        
        self.X = tf.keras.layers.Input(shape=(500,500,3), batch_size=self.batch_size,name='input_data',dtype='float32')
        self.y = tf.placeholder(tf.float32, shape=[None, 120,1], name='correct_labels')
        self.y_pred = tf.placeholder(tf.float32, shape=[None,120,1], name='predicted_labels')
        
        #get pretrained neural network
        #self.model = self.get_inception_v3()
        self.base_model = self.get_inception_resnet_v2()
        

        #update the input and output layers of the model
        #self.update_output_layer()
        self.model = tf.keras.Model(inputs=self.base_model.input, outputs=self.generate_output_layer())
        

        #access the generator for use during any training/testing sess
        self.gen = generator.generator(self.batch_size)
        
        #build model
        #if(self.train):
        if(self.i == 0): #if on the first iteration of training
            self.initial_train() #train the last nodes on the new output classes

        self.train_model() #train the entire model

        #evaulate the model
        #self.evaluate_model()

        #test a single image
        #self.use_model()

        #save the model
        #self.save_model()

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
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='rmsprop')        
        self.model.fit_generator(self.gen.generate_training_data(), steps_per_epoch=3000, epochs=3) #up steps p epoch to 3k and epochs to ~4

        for layer in self.model.layers[:780]:
            layer.trainable = False
        for layer in self.model.layers[780:]: #let's try only training the newly created final layers and leave the rest of
            #inception resnet untouched
            layer.trainable = True

        return True


    
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
            writer = tf.summary.FileWriter("../Reports/log/", sess.graph)
            sess.run(tf.global_variables_initializer())

            #initialize the main layers
            #x = self.input_layer()
            #y = self.output_layer()
            #y_pred = self.prediction_layer()
            
            #train using the train op
            # for i in range(1000):
            #    sess.run(train_op)
            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = False)
  
            #from tensorflow.keras.optimizers import SGD
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), options=run_opts)

            #save checkpoints of the model during training...
            #self.i = 0
            #del self.model
            #self.model = tf.keras.models.load_model('../models/model_ckpt_6.h5')

            while(self.i<100): #each 'i' will be 1 epoch, 100 iterations in total
                            

                #print("Running training for iterations: ", i*10, " to ", i*10+9)
                print("Running training for epoch: ", self.i)
                self.model.fit_generator(self.gen.generate_training_data(), steps_per_epoch=3000, epochs=1)
                #self.model.fit_generator(self.gen.generate_training_data(), steps_per_epoch=2, epochs=1)

                file_name = '../models/model_ckpt_' + str(self.i) + '.h5'
                self.model.save(file_name)
                i+=1


            #self.model.fit_generator(self.gen.generate_training_data(), steps_per_epoch=2, epochs=1)

            #print("evaluating... \n\n")            
            #self.model.evaluate_generator(self.gen.generate_testing_data(), steps=4290)
            #self.model.evaluate_generator(self.gen.generate_testing_data(), steps=1)
##
##            print("testing on a single batch... ")
##            #print a single prediction as well as the expected prediction
##            #X = tf.placeholder(tf.float32, shape=([1, 500,500,3]))
##            X,y = next(self.gen.generate_testing_data())
##            #X = [X[0],X[1]]
##            #X = np.vstack(X)
##            
##            #X = np.expand_dims(X, axis=0)
##            classification = self.model.predict(X, batch_size=2)
##            print("classifications: ",classification)
##            print("Actual Classification part 1: ", y[0])
##            print("\nActual Classficiation part 2: ", y[1])
##
            self.model.save('../models/model.h5')
            writer.close()
                
        #img_data = train_input_fn(index=index, data_amnt=batch_size)
        #breed_data = train_output_fn(index=index, data_amnt=batch_size)

       

    def evaluate_model(self):

        print("about to start testing :D")
        del self.model
        self.model = tf.keras.models.load_model('../models/model_ckpt_6.h5')
        total_correct_predictions = 0
        i = 0
        while(i< 100):
            X,y = next(self.gen.generate_training_data())
            classification = self.model.predict(X, batch_size=4) #get a batch of predictions

            #check the accuracy of each member of the batch
            j = 0
            while(j < 4):
                pred_classification_top, pred_classification_second, pred_classification_third = utils.get_breed_from_output(classification[j])
                actual_classification, temp, temp2 = utils.get_breed_from_output(y[0]) #this function gives 3 outputs, but in the actual breed output only the first matters
                #check if the prediction was correct
                if(pred_classification_top == actual_classification):
                    total_correct_predictions += 1
                elif(pred_classification_second == actual_classification):
                    total_correct_predictions += 0
                elif(pred_classification_third == actual_classification):
                    total_correct_predictions += 0               
                j+=1
            i+=1 #run another batch                                                                           

        print("Accuracy = ", total_correct_predictions/400)            

         
        return True

    def use_model(self, X=None):
        print("test text to see if file is being updated at save")
        if(self.load_model):
            del self.model
            self.model = tf.keras.models.load_model('../models/model_ckpt_2.h5')

        #print("testing on a single batch... ")
        #print a single prediction as well as the expected prediction
        #X = tf.placeholder(tf.float32, shape=([1, 500,500,3]))

        #test on training data to check if neural net is overfit
        X,y = next(self.gen.generate_training_data())
        #X = [X[0],X[1]]
        #X = np.vstack(X)
        
        #X = np.expand_dims(X, axis=0)
        classification = self.model.predict(X, batch_size=4)
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
    

import os
import os
import numpy as np #linear algebra
import pandas as pd #data preprocessing
import matplotlib.pyplot as plt #data visualization
import h5py
import PIL

import data.generator as generator #geneartes a single batch of data

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

    def prediction_layer(self):
        y_pred = tf.placeholder(tf.float32, shape=[None,120,1], name='predicted_labels')
        return y_pred

    def train_input_fn(index=0, data_amnt=1):
        input_img_data = dataFrameBuilder(data_amount=data_amnt,
                                          start_index=index)

        input_img_data = np.asarray(input_img_data)

        return input_img_data
    
    def get_inception_v3(self):
        inception_v3 = tf.keras.applications.InceptionV3(include_top=False,
                                                input_tensor=self.input_layer,
                                                classes=120)

        return inception_v3

    def train_op(self):
        return None

    def update_output_layer(self):
        #steps for adding a new output layer
        output_layer = self.inception_v3.output
        output_layer = tf.keras.layers.GlobalAveragePooling2D()(output_layer) #replace the current global avg pool 2d
        output_layer = tf.keras.layers.Dense(1024, activation='relu')(output_layer) 
        predictions = tf.keras.layers.Dense(120, activation='softmax')(output_layer) #120 classes in the new model
        self.inception_v3 = tf.keras.Model(inputs=inception_v3.input, outputs=predictions)
        return model
    
    def train_model(self):
        self.model = tf.keras.Model(inputs=inception_v3.input, output=self.predictions)

        #training configurations that worked on my computer
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.40
        config.gpu_options.allow_growth = True

        tf.reset_default_graph()

        with tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True)) as sess:
              writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

              #initialize the main layers
              x = self.input_layer()
              y = self.output_layer()
              y_pred = self.prediction_layer()

              self.update_output_layer()
              
              for i in range(1000):
                  sess.run(train_op)
                writer.close()
                

        self.inception_v3.compile(loss=tf.keras.losses.categorical_crossentropy,
                            optimizer='sgd')

        model.fit_generator(generator(batch_size), steps_per_epoch=10, epochs=50)

        img_data = train_input_fn(index=index, data_amnt=batch_size)
        breed_data = train_output_fn(index=index, data_amnt=batch_size)

       

    def evaluate_model(self):
        return model.evaluate(x=test_img_data, y=test_breed_data,batch_size=batch_size)
    
    def save_model(self):
        self.inception_v3.save('../../models/model.h5')
        return True
        
    def load_model(self):
        self.inception_v3 = keras.models.load_model('../..models/model.h5')
        print(self.inectpion_v3.summary())
        return True
    

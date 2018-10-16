import os
import numpy as np #linear algebra
import pandas as pd #data preprocessing
#import matplotlib.pyplot as plt #data visualization
import h5py
import PIL


import tensorflow as tf

import data.generator as generator #geneartes a single batch of data

gen = generator.generator(4)

X = tf.placeholder(tf.float32, shape=([1, 500,500,3]))

with tf.Session() as sess:
        model = tf.keras.models.load_model('../models/model_ckpt_4.h5')

        print("testing on a single batch... ")
        #print a single prediction as well as the expected prediction
        #X = tf.placeholder(tf.float32, shape=([1, 500,500,3]))
        X,y = next(gen.generate_testing_data())
        #X = [X[0],X[1]]
        #X = np.vstack(X)
        
        #X = np.expand_dims(X, axis=0)
        classification = model.predict(X, batch_size=4)
        print("classifications: ",classification)
        print("Actual Classification part 1: ", y[0])
        print("\nActual Classficiation part 2: ", y[1])


from __future__ import absolute_import, division, print_function

import random
import time

import tensorflow as tf
import numpy as np

import pathlib

tf.enable_eager_execution()

DEFAULT_IMAGE_PATH = 'E:\Dog_Breed_data\Images'
AUTOTUNE = tf.data.experimental.AUTOTUNE

#Build the dataset based on the Default image path 
def database_builder():
    image_dir = DEFAULT_IMAGE_PATH
    data_root = pathlib.Path(image_dir)
    
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    
    #iterate through all folder names to get labels and remove numbers
    label_names = sorted(item.name.split('-')[1] for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name.split('-')[1]] for path in all_image_paths]
    
    #build the database
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds

#resize an image and return it
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0 #normalize to [0,1] range -- dependent upon network being used

    return image

#loads the image into memory and preprocesses the image
def load_and_preprocess_image(path):
    image = tf.read_file(path)

    return preprocess_image(image)

def get_image_count():
    data_root = pathlib.Path(DEFAULT_IMAGE_PATH)
    all_image_paths = list(data_root.glob('*/*'))
    image_count = len(all_image_paths)

    return image_count

def change_range(image,label):
  return 2*image-1, label




BATCH_SIZE = 64

#prepare the dataset
ds = database_builder()
ds = ds.shuffle(buffer_size=(int(get_image_count()/2)))
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
keras_ds = ds.map(change_range)

steps_per_epoch=tf.ceil(get_image_count()/BATCH_SIZE).numpy()


#Fetch mobile_net and alter it to fit this project
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3), include_top=False)
mobile_net.trainable=False

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(120)]) #120 dog breeds

#training here
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)



logit_batch = model(image_batch).numpy()

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()

steps_per_epoch=tf.ceil(get_image_count()/BATCH_SIZE).numpy()

model.fit(ds, epochs=1, steps_per_epoch=3)

model.save('../models/dog_vision.h5')

#used for measuring the performance of our datasets
def timeit(ds, batches=2*steps_per_epoch+1):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(batches+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(batches, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
  print("Total time: {}s".format(end-overall_start))

#timeit(ds, batches=2*steps_per_epoch+1)

import models.inception_dog_breed_classifier as inception
import data.generator as generator
import numpy as np
import data.pre_process_images as pre_processor




batch_size = 10
def run():
    model = inception.inception_classifier()

def gen_test():
    gen = generator.generator(10)
    X,y = next(gen.generate_training_data())
    #Y = np.array(X)
    #print("Y.dimensions(): ", np.shape(X))

def pre_process():
    np.set_printoptions(threshold=np.nan) #numpy arrays will print the entire array now
    pre_processor.pre_process_images()

    

def main_old():
   pre_process()



run()

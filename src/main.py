import models.inception_dog_breed_classifier as inception
import data.generator as generator
import numpy as np

batch_size = 10
def run():
    model = inception.inception_classifier()

def gen_test():
    gen = generator.generator(10)
    X,y = next(gen.generate_training_data())
    #Y = np.array(X)
    #print("Y.dimensions(): ", np.shape(X))
    

run()

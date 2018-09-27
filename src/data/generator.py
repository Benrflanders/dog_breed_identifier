import math

#generates data for direct use with the main model builder script

class generator():
    def __init__(self, batch_size, features, number_of_images=20580):
        self.batch_size = batch_size

        self.labels_list = '../../data/processed/labels.csv'
        self.train_dir = '../../data/processed/train_images/'
        self.test_dir = '../../data/processed/test_images/'

        self.split = split #percentage of data that is the training set
        self.start = 0
        self.curr = 0
        self.end = number_of_images

        self.batch_features = np.zeros((self.batch_size, 500, 500, 3))
        self.batch_labels = np.zeros((batch_size,120))

    def next(self):
        self.curr+=1
        self.get_data()

    def get_training_data(self):
        #get from train_list.mat
        

        return True

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
                y[i] = #get the label of the current input data
            yield X, y

    def get_input_data(self, index=0, data_amnt=1):
        input_img_data = np.asarray(input_img_data)
        return input_img_data

    def get_test_input_data(self):

        return input_img_data

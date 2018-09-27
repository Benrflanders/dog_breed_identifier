import os
import sys
sys.path.append('..')
from PIL import Image, ImageOps
from scipy.io import loadmat

from src.utils.general_utils import *


'''
Converts all images in the "raw" images folder to 500 x 500 images and saves them
to the processed images folder. Images are either saved to train_images/ or
test_images/ based upon their location in the 
'''
def pre_process_images():
    os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
    raw_img_dir = '../data/raw/Images/'

    train_images_dir = '../data/processed/train_images/'
    test_images_dir = '../data/processed/test_images/'

    train_list = '../data/raw/train_list.mat'
    test_list = '../data/raw/test_list.mat'

    file_count = 0
    
    #for each folder
    raw_folders_list = os.listdir(raw_img_dir)
    print(raw_folders_list)
    for folder in raw_folders_list:
        if(folder[0] != '.'):
            curr_dir = raw_img_dir + folder
            print(curr_dir)
        
            print(os.listdir(curr_dir))
            curr_sub_folder = os.listdir(curr_dir)
            for file in curr_sub_folder:
                full_filename = curr_dir + '/' + file
                filename = full_filename.split('/')[4:]
                filename = filename[0] + '/' + filename[1]
                if(is_training_data(filename)):
                    output_dir = train_images_dir
                    normalize_image(full_filename, curr_dir, output_dir=output_dir)
                    file_count+= 1

                elif(is_testing_data(filename)):
                    output_dir = test_images_dir
                    normalize_image(full_filename, curr_dir, output_dir=output_dir)
                    file_count += 1

                else:
                    #print(filename)

                    print("PROPER FOLDER NOT FOUND")

                print(file_count)
                #print(file) #a single piece of data
                
                
    print(file_count)
    #convert each image to a 500 by 500 image using scaling
    #save to the proper folder: either train or test data folder

#convert all images to grayscale vectors-- no longer used as NN will be using 500 x 500 x 3 images as input
def normalize_image(file_name, original_dir,output_dir="../data/interim/"):
    os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')

    file = file_name #store the file name and location  
    image = Image.open(file)
    
    #scale the image to 500 x 500 x 3
    
    
    #  if width is under 500 add padding to make the image 500 x 500
    #  if height is under 500 add padding to make the image 500 x 500
    #  if either dimension > 500 scale down to get one dimension == to 500
    #     then add padding to get the other dimension == to 500
    
    width, height = image.size
    
    if(width > 500 or height > 500): #if width is greater than 500 or the height is greater than 500
        if(width > height): #determine which side length is greater
            image_ratio = 500/width #determine ratio of old image to new image
        else:
            image_ratio = 500/height
             
        width = int(width * image_ratio) #calculate new width
        height = int(height * image_ratio) #calculate new height
        image = image.resize((500, height), resample=1) #scale image to 
    
        width, height = image.size #refresh the width and height to make sure everything is still accurate
        delta_h = 500 - height #calculate the amount of height padding
        delta_w = 500 - width #calculate the amount of width padding -- should be 0
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        image = ImageOps.expand(image, padding)
    
        #print("test1... " + file_name) 
    
    
    width, height = image.size
    #after neither size is greater than 500 or already scaled
    if(width < 500 or height < 500): #if either dimension is less than 500 padding is added
        #add padding to the height and width to change the image dimensions to 500 x 500 x 3
        width, height = image.size #refresh the width and height to make sure everything is still accurate
        delta_h = 500 - height #calculate the amount of height padding
        delta_w = 500 - width #calculate the amount of width padding -- should be 0
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        image = ImageOps.expand(image, padding)
    
    
    image_id = file.split('/')
    image_id = image_id[len(image_id)-1]
    image_id = image_id.split('.')[0]

    width, height = image.size
    if(width != 500 or height != 500):
        print("something went wrong")
    
    #save the image to the output_dir with the same id
    image.save((output_dir + "" + image_id + ".png"),format='PNG')    
        
    return (file_name + " ... saved")


def is_training_data(filename):
    #get the list of all training dat
    os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
    train_list = '../data/raw/train_list.mat'
    mat = loadmat(train_list)
    for file in mat.get('file_list'):
        if(filename == file[0][0]):
          return True
        # else:
        #     import time
        #     print(filename)
        #     print(file[0][0])
        #     time.sleep(10)
    return False

def is_testing_data(filename):
    os.chdir('/Users/benflanders/Documents/github/kaggle_dog_breed_identifier/src')
    test_list = '../data/raw/test_list.mat'
    mat = loadmat(test_list)
    for file in mat.get('file_list'):
        if(filename == file[0][0]):
            return True

    return False
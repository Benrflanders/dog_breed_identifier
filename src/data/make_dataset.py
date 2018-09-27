import os

import matplotlib.pyplot as plt
import matplotlib.mlab as mlb
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd

from sklearn import preprocessing
from skimage import io, filters, morphology, util

from utils.general_utils import get_imgMatrix_from_id, get_random_id, get_id_from_filename, get_breed_value_from_id, populate_breeds, get_label_array_from_id, get_random_id
from src.data.data_downloader import download_data




def main():
    np.set_printoptions(threshold=np.nan) #numpy arrays will print the entire array now

    #please download the data for the project according to the README.md
    #download_data()

    pre_process_images()

    


if __name__ == "__main__":
    main()

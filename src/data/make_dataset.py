import os

import numpy as np
import data.pre_process_images as pre_processor


def main():
    np.set_printoptions(threshold=np.nan) #numpy arrays will print the entire array now

    #please download the data for the project according to the README.md
    #download_data()

    pre_processor.pre_process_images()

    


if __name__ == "__main__":
    main()

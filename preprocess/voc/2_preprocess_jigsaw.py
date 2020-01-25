import os
from os import listdir, mkdir
from os.path import isfile, join, isdir

import numpy as np
import imageio
# from scipy.misc import imsave
# from PIL import Image
# from random import randint
# import time
# from scipy.stats import mode
import cv2
# import gc
# import shutil 

import config as config

def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    return image_paths

#get '17asdfasdf2d_0_0.jpg' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'
def basename(path):
	return path.split('/')[-1]

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

def generate_rotated_images(input_folder, output_folder, rotations):
    image_paths = get_image_paths(input_folder)
    for image_path in image_paths:
        print(image_path)
        image = cv2.imread(image_path)

        #zero rotation
        label = str(0)
        image_output_folder = join(output_folder, label)
        confirm_output_folder(image_output_folder)
        output_path = join(image_output_folder, basename(image_path))
        imageio.imwrite(output_path, image)

        for rotation in rotations:
            label = str(rotation)
            rotation_times = int(rotation/90)
            rotated_image = np.rot90(image, rotation_times)
            image_output_folder = join(output_folder, label)
            confirm_output_folder(image_output_folder)
            output_path = join(image_output_folder, basename(image_path))
            imageio.imwrite(output_path, rotated_image)

    print(len(image_paths))

if __name__ == "__main__":

    rotations = [90, 180, 270]
    generate_rotated_images(config.output_folder_train_full, config.trainval_folder_full_train, rotations)
    generate_rotated_images(config.output_folder_val_full, config.trainval_folder_full_val, rotations)
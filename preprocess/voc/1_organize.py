import os
from os import listdir
from os.path import isfile, join, isdir

import numpy as np
# from scipy.misc import imsave
# from PIL import Image
from random import randint
import time
from scipy.stats import mode
# import cv2
import gc
import shutil 

import config as config

def basename(path):
	return path.split('/')[-1]

def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    return image_paths

def get_image_list_from_annotation_txt_path(annotation_txt_path, images_path):
    lines = open(annotation_txt_path, 'r').readlines()
    images = [line.split(" ")[0] for line in lines if ' 1' in line]
    image_paths = [join(images_path, x) + '.jpg' for x in images]
    return image_paths

def move_images_to_folder(annotation_txt_paths, output_folder):
    for annotation_txt_path in annotation_txt_paths:
        image_paths = get_image_list_from_annotation_txt_path(annotation_txt_path, images_path)
        for image_path in image_paths:
            image_out_path = join(output_folder, basename(image_path))
            shutil.copyfile(image_path, image_out_path)

if __name__ == "__main__":

    images_path = join(config.master_folder, config.images_folder)

    image_sets_path = join(config.master_folder, config.image_sets_folder)
    annotation_txt_paths = get_image_paths(image_sets_path)
    annotation_txt_paths_train = [x for x in annotation_txt_paths if '_train.' in x]
    annotation_txt_paths_val = [x for x in annotation_txt_paths if '_val.' in x]

    move_images_to_folder(annotation_txt_paths_train, config.output_folder_train_full)
    move_images_to_folder(annotation_txt_paths_val, config.output_folder_val_full)
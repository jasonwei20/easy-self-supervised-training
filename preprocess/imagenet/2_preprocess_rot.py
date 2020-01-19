from pathlib import Path
from multiprocessing import Pool

import numpy as np
import imageio
import cv2

input_folder = Path("/home/brenta/scratch/data/ImageNet_one_folder/train")
output_folder = Path("/home/brenta/scratch/data/imagenet_rotnet/train")
rotations = [90, 180, 270]

def get_chunks(lst, n):
    """Yield n successive chunks from lst."""
    size = int(len(lst) / n)
    output_list = []
    for i in range(0, n):
        sub_list = lst[i*size:i*size + size]
        output_list.append(sub_list)
    if len(lst) % n != 0:
        for i in range((n-1)*size+1, len(lst)):
            output_list[-1].append(lst[i])
    return output_list

def confirm_output_folder(folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)

def get_subfolders(master_folder_path):
    return [x for x in master_folder_path.iterdir() if x.is_dir()]

def get_image_paths(subfolder):
    return [x for x in subfolder.iterdir() if x.is_file()]

def generate_rotated_images(image_paths):

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        #zero rotation
        label = str(0)
        image_output_folder = output_folder.joinpath(label)
        confirm_output_folder(image_output_folder)
        output_path = image_output_folder.joinpath(image_path.name)
        imageio.imwrite(output_path, image)
        # print(output_path)

        for rotation in rotations:
            label = str(rotation)
            rotation_times = int(rotation/90)
            rotated_image = np.rot90(image, rotation_times)
            image_output_folder = output_folder.joinpath(label)
            confirm_output_folder(image_output_folder)
            output_path = image_output_folder.joinpath(image_path.name)
            imageio.imwrite(output_path, rotated_image)
            # print(output_path)

    print(len(image_paths))

def gen_training_data_rot():
    image_paths = get_image_paths(input_folder)
    # image_paths = image_paths[:96]

    #single thread
    # generate_rotated_images(image_paths)

    #multi-thread
    chunks = get_chunks(image_paths, 8)
    for chunk in chunks:
        print(len(chunk))
    p = Pool(8)
    p.map(generate_rotated_images, chunks)

if __name__ == "__main__":

    gen_training_data_rot()
    # generate_rotated_images(config.output_folder_val_full, config.trainval_folder_full_val, rotations)
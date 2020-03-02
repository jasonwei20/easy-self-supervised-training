
from pathlib import Path
import shutil 
import random

mags_file = 'mags_resnet18_imagenet.csv'
train_folder = Path('/home/brenta/scratch/data/imagenet_rotnet/train/')

def read_mags_file(mags_file):
    lines = open(mags_file, "r").readlines()[1:]#[1:250]
    tup_list = []
    for line in lines:
        parts = line.replace('\n', '').split(',')
        image_name = parts[0]
        mag_grad = float(parts[2])
        tup = (image_name, mag_grad)
        tup_list.append(tup)

    random.shuffle(tup_list)
    num_val = 10000
    tup_list_val = tup_list[:num_val]
    tup_list_train = tup_list[num_val:]
    
    ordered_tup_list_train = sorted(tup_list_train, key=lambda x: x[1])
    ordered_tup_list_train = [x[0] for x in ordered_tup_list_train]
    ordered_tup_list_val = sorted(tup_list_val, key=lambda x: x[1])
    ordered_tup_list_val = [x[0] for x in ordered_tup_list_val]

    return ordered_tup_list_train, ordered_tup_list_val

def copy_images(image_names, source_folder, target_folder):
    for image_name in image_names:
        source_path = source_folder.joinpath(image_name)
        target_path = target_folder.joinpath(image_name.replace('/', '_'))
        print(source_path, target_path)
        shutil.copyfile(source_path, target_path)

if __name__ == "__main__":
    ordered_tup_list_train, ordered_tup_list_val = read_mags_file(mags_file)
    low_grad_images_train = ordered_tup_list_train[:int(len(ordered_tup_list_train)/2)]
    high_grad_images_train = ordered_tup_list_train[int(len(ordered_tup_list_train)/2):]
    low_grad_images_val = ordered_tup_list_val[:int(len(ordered_tup_list_val)/2)]
    high_grad_images_val = ordered_tup_list_val[int(len(ordered_tup_list_val)/2):]

    copy_images(low_grad_images_train, train_folder, Path('/home/brenta/scratch/jason/data/imagenet/grad/first_test/train/0'))
    copy_images(high_grad_images_train, train_folder, Path('/home/brenta/scratch/jason/data/imagenet/grad/first_test/train/1'))
    copy_images(low_grad_images_val, train_folder, Path('/home/brenta/scratch/jason/data/imagenet/grad/first_test/val/0'))
    copy_images(high_grad_images_val, train_folder, Path('/home/brenta/scratch/jason/data/imagenet/grad/first_test/val/1'))
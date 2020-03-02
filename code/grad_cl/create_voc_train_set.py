
from pathlib import Path
import shutil 
import random

mags_file = '/home/brenta/scratch/jason/logs/voc/vanilla/exp_45/resnet18_e2_mb120_va0.48665.pt_grads.csv'
input_folder = '/home/brenta/scratch/jason/data/voc/voc_trainval_full/train/'
output_train_folder = '/home/brenta/scratch/jason/data/voc/cl_ranked/reverse_train_resnet18_e2_mb120_va0.48665/train/'

def read_mags_file(mags_file):
    lines = open(mags_file, "r").readlines()[1:]#[1:250]
    tup_list = []
    for line in lines:
        parts = line.replace('\n', '').split(',')
        image_name = parts[0]
        mag_grad = float(parts[2])
        tup = (image_name, mag_grad)
        tup_list.append(tup)
    tup_list = sorted(tup_list, key=lambda x: x[1])
    tup_list = reversed(tup_list)
    return tup_list

def append_zeros(s, target_len):
    while len(s) < target_len:
        s = "0" + s 
    return s

def create_new_train_folder(tup_list, output_train_folder):

    for i, tup in enumerate(tup_list):
        source_name = tup[0]
        target_name = source_name.replace('/', '/' + append_zeros(str(i), 5) + '_')
        source_path = input_folder + source_name
        target_path = output_train_folder + target_name
        shutil.copyfile(source_path, target_path)
        print(i, source_path, target_path)

if __name__ == "__main__":
    tup_list = read_mags_file(mags_file)
    create_new_train_folder(tup_list, output_train_folder)
from pathlib import Path
import shutil
import random
from multiprocessing import Pool
random.seed(42)

master_folder_path = Path("/home/brenta/scratch/data/ImageNet_full/train/")
output_master_folder_path = Path("/home/brenta/scratch/data/ImageNet_one_folder/train")

def get_chunks(lst, n):
    """Yield n successive chunks from lst."""
    size = int(len(lst) / n)
    output_list = []
    for i in range(0, len(lst), size):
        output_list.append(lst[i:i + size])
    return output_list

def confirm_output_folder(folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)

def get_subfolders(master_folder_path):
    return [x for x in master_folder_path.iterdir() if x.is_dir()]

def get_image_paths(subfolder):
    return [x for x in subfolder.iterdir() if x.is_file()]

def copy_subfolder(subfolder):

    output_subfolder = output_master_folder_path
    confirm_output_folder(output_subfolder)

    image_paths = get_image_paths(subfolder)
    for image_path in image_paths:
        output_image_path = output_subfolder.joinpath(image_path.name)
        shutil.copyfile(image_path, output_image_path)
        # print(image_path, output_image_path)

    print(f"Finished {subfolder}")

def copy_subfolders(subfolders):
    for subfolder in subfolders:
        copy_subfolder(subfolder)

def copy_image_into_one_folder():

    subfolders = get_subfolders(master_folder_path)

    # single thread
    # copy_subfolders(subfolders)

    #multi-thread
    chunks = get_chunks(subfolders, 8)
    p = Pool(8)
    p.map(copy_subfolders, chunks)
        

if __name__ == "__main__":

    copy_image_into_one_folder()
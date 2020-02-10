from pathlib import Path
import shutil
import random
from multiprocessing import Pool

master_folder_path = Path("/home/brenta/scratch/data/imagenet_rotnet/val/")
output_master_paths = [ Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_15/val"), 
                        Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_16/val"), 
                        Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_17/val"), 
                        Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_18/val"), 
                        Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_19/val"), 
                        Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_20/val"),]

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

# def copy_subfolder(subfolder):

#     output_subfolder = output_master_folder_path.joinpath(subfolder.name)
#     confirm_output_folder(output_subfolder)

#     image_paths = get_image_paths(subfolder)
#     # random.shuffle(image_paths)
#     for image_path in image_paths[:k_per_class]:
#         output_image_path = output_subfolder.joinpath(image_path.name)
#         shutil.copyfile(image_path, output_image_path)
#         # print(image_path, output_image_path)
    
#     print(f"Finished {subfolder}")

def copy_subfolders(subfolders):
    for subfolder in subfolders:
        copy_subfolder(subfolder)

def copy_files(file_paths):
    for f in file_paths:
        file_ending = "/".join(str(f).split("/")[-2:])

        for output_master_path in output_master_paths:
            dest = output_master_path.joinpath(file_ending)
            shutil.copyfile(str(f), str(dest))

def copy_image_subset():

    subfolders = get_subfolders(master_folder_path)

    all_paths = []
    for subfolder in subfolders:
        paths = get_image_paths(subfolder)
        all_paths += paths

    # single thread
    # copy_subfolders(subfolders)

    #multi-thread
    chunks = get_chunks(all_paths, 8)
    for chunk in chunks:
        print(len(chunk))
    p = Pool(8)
    p.map(copy_files, chunks)
        

if __name__ == "__main__":

    for rot in ['0', '90', '180', '270']:
        for master_folder in output_master_paths:
            confirm_output_folder(master_folder.joinpath(rot))

    copy_image_subset()
    print("finished")
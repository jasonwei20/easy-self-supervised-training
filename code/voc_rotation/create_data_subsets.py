from pathlib import Path
from utils import *
import shutil

train_data_path = Path('data/voc_trainval_full/train')

def get_easiest_percent(paths_and_data, fraction):

    sorted_paths_and_data = sorted(paths_and_data, key=lambda x: x[1])
    num_to_return = int(fraction*len(sorted_paths_and_data))
    sorted_paths_and_data_subset = sorted_paths_and_data[-num_to_return:]
    return [x[0] for x in sorted_paths_and_data_subset]

def get_hardest_percent(paths_and_data, fraction):

    sorted_paths_and_data = sorted(paths_and_data, key=lambda x: x[1])
    num_to_return = int(fraction*len(sorted_paths_and_data))
    sorted_paths_and_data_subset = sorted_paths_and_data[:num_to_return]
    return [x[0] for x in sorted_paths_and_data_subset]

def get_correct(paths_and_data):
    return [x[0] for x in paths_and_data if x[2] == "correct"]

def get_incorrect(paths_and_data):
    return [x[0] for x in paths_and_data if x[2] == "incorrect"]

def get_paths_and_data(csv_folder):
    csv_list = get_image_paths(csv_folder)
    paths_and_data = []

    for csv in csv_list:
        lines = open(csv, 'r').readlines()
        for line in lines[1:]:
            parts = line[:-1].split(",")
            image_name = parts[0]
            ground_truth = parts[1]
            pred = parts[2]
            conf = float(parts[3])
            image_name_and_class = '/'.join([ str(csv).split(".")[-2].split('/')[-1], image_name ])
            if ground_truth == pred:
                paths_and_data.append((image_name_and_class, conf, "correct"))
            else:
                paths_and_data.append((image_name_and_class, conf, "incorrect"))

    return paths_and_data

def move_into_new_folder(paths_list, train_data_path, new_folder):
    print("moving", len(paths_list), "files")
    for path in paths_list:
        source = train_data_path.joinpath(path)
        destination = Path(new_folder).joinpath(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source), str(destination))
    val_source = train_data_path.parent.joinpath("val")
    val_dest = Path(new_folder).parent.joinpath("val")
    # val_dest.mkdir(parents=True, exist_ok=True)
    print(val_source, val_dest)
    shutil.copytree(str(val_source), str(val_dest))

if __name__ == "__main__":

    csv_folder = Path("outputs/resnet18_e30_va0.68715_train")
    paths_and_data = get_paths_and_data(csv_folder)

    easiest_paths_tenth = get_easiest_percent(paths_and_data, 0.1)
    move_into_new_folder(easiest_paths_tenth, train_data_path, "data/voc_trainval_easy_0.1/train")

    easiest_paths_quarter = get_easiest_percent(paths_and_data, 0.25)
    move_into_new_folder(easiest_paths_quarter, train_data_path, "data/voc_trainval_easy_0.25/train")

    hardest_paths_tenth = get_hardest_percent(paths_and_data, 0.1)
    move_into_new_folder(hardest_paths_tenth, train_data_path, "data/voc_trainval_hard_0.1/train")

    hardest_paths_quarter = get_hardest_percent(paths_and_data, 0.25)
    move_into_new_folder(hardest_paths_quarter, train_data_path, "data/voc_trainval_hard_0.25/train")

    correct_paths = get_correct(paths_and_data)
    move_into_new_folder(correct_paths, train_data_path, "data/voc_trainval_correct/train")

    incorrect_paths = get_incorrect(paths_and_data)
    move_into_new_folder(incorrect_paths, train_data_path, "data/voc_trainval_incorrect/train")
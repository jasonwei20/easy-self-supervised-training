from multiprocessing import Pool
from pathlib import Path
from random import sample
import shutil 

hard_frac = 0.1

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

def training_preds_file_to_tuple_list(training_preds_file):
    
    tuple_list = []
    name_list = []

    lines = open(str(training_preds_file), "r").readlines()
    for line in lines[1:]:
        parts = line[:-1].split(",")
        name = parts[0]
        gt = name.split("/")[0]
        pred = parts[1]
        conf = float(parts[2])
        tuple_list.append((name, conf))
        name_list.append(name)
    
    return name_list, sorted(tuple_list, key=lambda x:x[1])

def get_random_sampling(random_sample_frac, hards, name_list):
    
    num_desired = int(random_sample_frac * len(hards))
    exclude_set = set(hards)
    candidates = [x for x in name_list if x not in exclude_set]
    random_samples = sample(candidates, num_desired)
    return random_samples

##### for new experiment of first tenth, second tenth, etc

def get_hards_specific(start_frac, end_frac, tuple_list):
    start_idx = int(start_frac * len(tuple_list))
    end_idx = int(end_frac * len(tuple_list))
    return [tup[0] for tup in tuple_list[start_idx:end_idx]]

if __name__ == "__main__":

    name_list, tuple_list = training_preds_file_to_tuple_list(training_preds_file)
    print("finished sorting tuple list")

    get_training_list_spec(0.0, 0.1, tuple_list, name_list, 1, train_folder_path)
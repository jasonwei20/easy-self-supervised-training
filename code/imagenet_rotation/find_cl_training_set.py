#first get predictions on the entire training set
#remove the predictions that are in the first half
#sort the predictions by confidence to get the top x percent
#uniformly sample from the remaining distribution
#put the predictions into a training folder

from multiprocessing import Pool
from pathlib import Path
from random import sample
import shutil 

training_order_file = Path("/home/brenta/scratch/jason/logs/imagenet/vanilla/exp_10/log_training_order_mb40000.csv")
training_preds_file = Path("/home/brenta/scratch/jason/outputs/image_net/vanilla/resnet18_e0_mb40000_va0.80428.pt/train_preds.csv")
training_folder = Path("/home/brenta/scratch/data/imagenet_rotnet/train")
hard_frac = 0.1

output_folder = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_35")

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

def get_training_order_list(training_order_file):
    training_order_list = []
    lines = open(str(training_order_file), "r").readlines()
    for line in lines:
        training_order_list.append(line[:-1])
    return training_order_list

def training_preds_file_to_tuple_list(training_preds_file):
    
    tuple_list = []

    lines = open(str(training_preds_file), "r").readlines()
    for line in lines[1:]:
        parts = line[:-1].split(",")
        name = parts[0]
        gt = name.split("/")[0]
        pred = parts[1]
        conf = float(parts[2])
        tuple_list.append((name, conf))
    
    return sorted(tuple_list, key=lambda x:x[1])

def get_hards(frac, tuple_list, training_order_list, from_earlier):

    if from_earlier: 
        exclude_set = set(training_order_list[int(len(training_order_list)/2):])
        early_tuple_list = [tup for tup in tuple_list if tup[0] not in exclude_set]
        idx = int(frac * len(early_tuple_list))
        return [tup[0] for tup in early_tuple_list[:idx]]
    else:
        idx = int(frac * len(tuple_list))
        return [tup[0] for tup in tuple_list[:idx]]

def get_random_sampling(random_sample_frac, hards, training_order_list):
    
    num_desired = int(random_sample_frac * len(hards))
    exclude_set = set(hards)
    candidates = [x for x in training_order_list if x not in exclude_set]
    random_samples = sample(candidates, num_desired)
    return random_samples

def move_samples(random_samples):
    for sample in random_samples:
        source = training_folder.joinpath(sample)
        dest = output_folder.joinpath('train').joinpath(sample)
        shutil.copyfile(source, dest)
    print("Done", len(random_samples))

def get_training_list(tuple_list, training_order_list, from_earlier, random_sample_frac, output_folder):
    hards = get_hards(hard_frac, tuple_list, training_order_list, from_earlier)
    random_samples = get_random_sampling(random_sample_frac, hards, training_order_list)
    samples = hards + random_samples
    print(len(samples), "samples to move")

    for _class in ['0', '90', '180', '270']:
        output_folder.joinpath("train").joinpath(_class).mkdir(parents=True, exist_ok=True)

    chunks = get_chunks(samples, 8)
    for chunk in chunks:
        print('ready', len(chunk))
    p = Pool(8)
    p.map(move_samples, chunks)

def calculate_frac_differences(frac, tuple_list, training_order_list):

    name_to_conf = {tup[0]:tup[1] for tup in tuple_list}
    training_order_list = training_order_list[:-10000]
    left_sum, right_sum = 0, 0
    left_names = training_order_list[:int(frac * len(training_order_list))]
    right_names = training_order_list[-int(frac * len(training_order_list)):]
    for name in left_names:
        left_sum += name_to_conf[name]
    
    for name in right_names:
        right_sum += name_to_conf[name]
    
    print(len(left_names), len(right_names))

    return left_sum / int(frac * len(training_order_list)), right_sum / int(frac * len(training_order_list)), int(frac * len(training_order_list))

##### for new experiment of first tenth, second tenth, etc

def get_hards_specific(start_frac, end_frac, tuple_list):
    start_idx = int(start_frac * len(tuple_list))
    end_idx = int(end_frac * len(tuple_list))
    return [tup[0] for tup in tuple_list[start_idx:end_idx]]

def get_training_list_spec(start_frac, end_frac, tuple_list, training_order_list, random_sample_frac, output_folder):
    hards = get_hards_specific(start_frac, end_frac, tuple_list)
    random_samples = get_random_sampling(random_sample_frac, hards, training_order_list)
    samples = hards + random_samples
    print(len(samples), "samples to move")

    for _class in ['0', '90', '180', '270']:
        output_folder.joinpath("train").joinpath(_class).mkdir(parents=True, exist_ok=True)

    chunks = get_chunks(samples, 8)
    for chunk in chunks:
        print('ready', len(chunk))
    p = Pool(8)
    p.map(move_samples, chunks)

if __name__ == "__main__":

    tuple_list = training_preds_file_to_tuple_list(training_preds_file)
    training_order_list = get_training_order_list(training_order_file)
    print("finished sorting tuple list")

    get_training_list_spec(0.9, 1.0, tuple_list, training_order_list, 1, output_folder)
    # print(calculate_frac_differences(1, tuple_list, training_order_list))
    # get_training_list(tuple_list, training_order_list, True, 1, output_folder)
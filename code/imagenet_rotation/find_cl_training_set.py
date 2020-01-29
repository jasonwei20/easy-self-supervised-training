#first get predictions on the entire training set
#remove the predictions that are in the first half
#sort the predictions by confidence to get the top x percent
#uniformly sample from the remaining distribution
#put the predictions into a training folder

from pathlib import Path

training_order_file = Path("/home/brenta/scratch/jason/logs/imagenet/vanilla/exp_10/log_training_order_mb40000.csv")
training_preds_file = Path("")
training_folder = Path("/home/brenta/scratch/data/imagenet_rotnet/train")

output_folder_15 = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_15")
output_folder_16 = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_16")
output_folder_17 = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_17")
output_folder_18 = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_18")
output_folder_19 = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_19")
output_folder_20 = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_20")

def get_hards(training_order_file, training_preds_file, from_earlier):
    return NotImplemented

def get_random_sampling(frac, )

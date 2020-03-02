import argparse
from pathlib import Path

import torch

from utils import (get_classes, get_log_csv_name, get_log_csv_train_order)

exp_num = 37

parser = argparse.ArgumentParser(
    description="DeepSlide",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

###############################################################
#               PROCESSING AND PATCH GENERATION               #
###############################################################
# This is the input for model training, automatically built.
parser.add_argument(
    "--train_folder",
    type=Path,
    default=Path("/home/brenta/scratch/data/imagenet_rotnet/"),
    help="Location of the automatically built training input folder")

parser.add_argument("--image_ext",
                    type=str,
                    default="jpg",
                    help="Image extension for saving patches")

########################################
#               TRAINING               #
########################################

# Name of checkpoint file to load from.
parser.add_argument(
    "--checkpoint_file",
    type=Path,
    default=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_10/resnet18_e0_mb40000_va0.80428.pt"), #"/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_10/resnet18_e0_mb40000_va0.80428.pt"
    help="Checkpoint file to load if resume_checkpoint_path is True")
parser.add_argument("--log_folder",
                    type=Path,
                    default=Path("/home/brenta/scratch/jason/logs/imagenet/vanilla/exp_" + str(exp_num)),
                    help="Directory to save logs to")

#######################################################
#               ARGUMENTS FROM ARGPARSE               #
#######################################################
args = parser.parse_args()

# Device to use for PyTorch code.


# Compute the mean and standard deviation for the given set of WSI for normalization.


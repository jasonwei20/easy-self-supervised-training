import argparse
from pathlib import Path

import torch

from compute_stats import compute_stats
from utils import (get_classes, get_log_csv_name)

# Source: https://stackoverflow.com/questions/12151306/argparse-way-to-include-default-values-in-help
parser = argparse.ArgumentParser(
    description="DeepSlide",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#######################################
#               GENERAL               #
#######################################
# Number of processes to use.
parser.add_argument("--num_workers",
                    type=int,
                    default=8,
                    help="Number of workers to use for IO")
# Default shape for ResNet in PyTorch.
parser.add_argument("--patch_size",
                    type=int,
                    default=224,
                    help="Size of the patches extracted from the WSI")

# # Where the CSV file labels will go.
# parser.add_argument("--labels_train",
#                     type=Path,
#                     default=Path("labels_train.csv"),
#                     help="Location to store the CSV file labels for training")
# parser.add_argument("--labels_val",
#                     type=Path,
#                     default=Path("labels_val.csv"),
#                     help="Location to store the CSV file labels for validation")
# parser.add_argument("--labels_test",
#                     type=Path,
#                     default=Path("labels_test.csv"),
#                     help="Location to store the CSV file labels for testing")

###############################################################
#               PROCESSING AND PATCH GENERATION               #
###############################################################
# This is the input for model training, automatically built.
parser.add_argument(
    "--train_folder",
    type=Path,
    # default=Path("/home/brenta/scratch/data/imagenet_rotnet/"),
    default=Path("/home/ifsdata/vlg/jason/easy-self-supervised-training/data/voc_trainval_full"),
    help="Location of the automatically built training input folder")

# Folders of patches by WSI in training set, used for finding training accuracy at WSI level.
# parser.add_argument(
#     "--patches_eval_train",
#     type=Path,
#     default=Path("patches_eval_train"),
#     help=
#     "Folders of patches by WSI in training set, used for finding training accuracy at WSI level"
# )
# # Folders of patches by WSI in validation set, used for finding validation accuracy at WSI level.
# parser.add_argument(
#     "--patches_eval_val",
#     type=Path,
#     default=Path("patches_eval_val"),
#     help=
#     "Folders of patches by WSI in validation set, used for finding validation accuracy at WSI level"
# )

parser.add_argument("--image_ext",
                    type=str,
                    default="JPEG",
                    help="Image extension for saving patches")

#########################################
#               TRANSFORM               #
#########################################
parser.add_argument(
    "--color_jitter_brightness",
    type=float,
    default=0,
    help=
    "Random brightness jitter to use in data augmentation for ColorJitter() transform"
)
parser.add_argument(
    "--color_jitter_contrast",
    type=float,
    default=0,
    help=
    "Random contrast jitter to use in data augmentation for ColorJitter() transform"
)
parser.add_argument(
    "--color_jitter_saturation",
    type=float,
    default=0,
    help=
    "Random saturation jitter to use in data augmentation for ColorJitter() transform"
)
parser.add_argument(
    "--color_jitter_hue",
    type=float,
    default=0,
    help=
    "Random hue jitter to use in data augmentation for ColorJitter() transform"
)

########################################
#               TRAINING               #
########################################
# Model hyperparameters.
parser.add_argument("--num_epochs",
                    type=int,
                    default=200,
                    help="Number of epochs for training")
# Choose from [18, 34, 50, 101, 152].
parser.add_argument(
    "--num_layers",
    type=int,
    default=18,
    help=
    "Number of layers to use in the ResNet model from [18, 34, 50, 101, 152]")
parser.add_argument("--learning_rate",
                    type=float,
                    default=0.003,
                    help="Learning rate to use for gradient descent")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="Mini-batch size to use for training")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-4,
                    help="Weight decay (L2 penalty) to use in optimizer")
parser.add_argument("--learning_rate_decay",
                    type=float,
                    default=0.85,
                    help="Learning rate decay amount per epoch")
parser.add_argument("--resume_checkpoint",
                    type=bool,
                    default=False,
                    help="Resume model from checkpoint file")
parser.add_argument("--save_interval",
                    type=int,
                    default=1,
                    help="Number of epochs between saving checkpoints")
# Where models are saved.
parser.add_argument("--checkpoints_folder",
                    type=Path,
                    default=Path("/home/ifsdata/vlg/data_jason/checkpoints/image_net/vanilla_train"),
                    help="Directory to save model checkpoints to")

# Name of checkpoint file to load from.
parser.add_argument(
    "--checkpoint_file",
    type=Path,
    default=Path("xyz.pt"),
    help="Checkpoint file to load if resume_checkpoint_path is True")
# ImageNet pretrain?
parser.add_argument("--pretrain",
                    type=bool,
                    default=False,
                    help="Use pretrained ResNet weights")
parser.add_argument("--log_folder",
                    type=Path,
                    default=Path("logs/imagenet/vanilla"),
                    help="Directory to save logs to")

##########################################
#               PREDICTION               #
##########################################
# Selecting the best model.
# Automatically select the model with the highest validation accuracy.
parser.add_argument(
    "--auto_select",
    type=bool,
    default=True,
    help="Automatically select the model with the highest validation accuracy")
# Where to put the training prediction CSV files.
parser.add_argument(
    "--preds_train",
    type=Path,
    default=Path("preds_train"),
    help="Directory for outputting training prediction CSV files")
# Where to put the validation prediction CSV files.
parser.add_argument(
    "--preds_val",
    type=Path,
    default=Path("preds_val"),
    help="Directory for outputting validation prediction CSV files")
# Where to put the testing prediction CSV files.
parser.add_argument(
    "--preds_test",
    type=Path,
    default=Path("preds_test"),
    help="Directory for outputting testing prediction CSV files")

#######################################################
#               ARGUMENTS FROM ARGPARSE               #
#######################################################
args = parser.parse_args()

# Device to use for PyTorch code.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Automatically read in the classes.
classes = get_classes(args.train_folder.joinpath("val"))
num_classes = len(classes)

# This is the input for model training, automatically built.
train_patches = args.train_folder.joinpath("train")
val_patches = args.train_folder.joinpath("val")

# Compute the mean and standard deviation for the given set of WSI for normalization.
path_mean, path_std = ( [0.40853017568588257, 0.4573926329612732, 0.48035722970962524], 
                        [0.28722450137138367, 0.27334490418434143, 0.2799932360649109])
                        #compute_stats(folderpath=args.train_folder.joinpath("train").joinpath("0"),
                                    #image_ext=args.image_ext)

# Only used is resume_checkpoint is True.
resume_checkpoint_path = args.checkpoints_folder.joinpath(args.checkpoint_file)

# Named with date and time.
log_csv = get_log_csv_name(log_folder=args.log_folder)

# Does nothing if auto_select is True.
eval_model = args.checkpoints_folder.joinpath(args.checkpoint_file)

# Print the configuration.
# Source: https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python/44689627
# chr(10) and chr(9) are ways of going around the f-string limitation of
# not allowing the '\' character inside.
print(f"###############     CONFIGURATION     ###############\n"
      f"{chr(10).join(f'{k}:{chr(9)}{v}' for k, v in vars(args).items())}\n"
      f"device:\t{device}\n"
      f"classes:\t{classes}\n"
      f"num_classes:\t{num_classes}\n"
      f"train_patches:\t{train_patches}\n"
      f"val_patches:\t{val_patches}\n"
      f"path_mean:\t{path_mean}\n"
      f"path_std:\t{path_std}\n"
      f"resume_checkpoint_path:\t{resume_checkpoint_path}\n"
      f"log_csv:\t{log_csv}\n"
      f"eval_model:\t{eval_model}\n"
    #   f"threshold_search:\t{threshold_search}\n"
    #   f"colors:\t{colors}\n"
      f"\n#####################################################\n\n\n")
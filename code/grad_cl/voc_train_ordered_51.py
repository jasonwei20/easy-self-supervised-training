import torch
from pathlib import Path

from utils import (get_classes, get_log_csv_name, get_log_csv_train_order)
from utils_model_ordered import train_resnet_ordered

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_mean, path_std = ( [0.40853017568588257, 0.4573926329612732, 0.48035722970962524], 
                        [0.28722450137138367, 0.27334490418434143, 0.2799932360649109])

exp_num = 51

# Named with date and time.
train_folder = Path("/home/brenta/scratch/jason/data/voc/cl_ranked/reverse_train_resnet18_e2_mb120_va0.48665")
log_folder = Path("/home/brenta/scratch/jason/logs/voc/vanilla/exp_" + str(exp_num))
log_csv = get_log_csv_name(log_folder=log_folder)
train_order_csv = get_log_csv_train_order(log_folder=log_folder)
classes = get_classes(train_folder.joinpath("train"))
num_classes = len(classes)

# Training the ResNet.
print("\n\n+++++ Running 3_train.py +++++")
train_resnet_ordered(batch_size=256,
             checkpoints_folder=Path("/home/brenta/scratch/jason/checkpoints/voc/vanilla/exp_" + str(exp_num)),
             classes=classes,
             color_jitter_brightness=0,
             color_jitter_contrast=0,
             color_jitter_hue=0,
             color_jitter_saturation=0,
             device=device,
             learning_rate=0.0001,
             learning_rate_decay=0.5,
             log_csv=log_csv,
             train_order_csv=train_order_csv,
             num_classes=num_classes,
             num_layers=18,
             num_workers=8,
             path_mean=path_mean,
             path_std=path_std,
             pretrain=False,
             resume_checkpoint=False,
             resume_checkpoint_path=None,
             save_interval=0,
             num_epochs=1,
             train_folder=train_folder,
             weight_decay=1e-4)
print("+++++ Finished running 3_train.py +++++\n\n")
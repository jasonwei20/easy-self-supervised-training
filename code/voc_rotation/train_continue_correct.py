import config
from pathlib import Path
from utils_model import train_resnet
from utils import (get_classes, get_log_csv_name)

# Training the ResNet.
print("\n\n+++++ Running 3_train.py +++++")
train_resnet(batch_size=config.args.batch_size,
             checkpoints_folder=Path('checkpoints_correct'),
             classes=config.classes,
             color_jitter_brightness=config.args.color_jitter_brightness,
             color_jitter_contrast=config.args.color_jitter_contrast,
             color_jitter_hue=config.args.color_jitter_hue,
             color_jitter_saturation=config.args.color_jitter_saturation,
             device=config.device,
             learning_rate=config.args.learning_rate,
             learning_rate_decay=config.args.learning_rate_decay,
             log_csv=get_log_csv_name(log_folder=Path('logs/correct')),
             num_classes=config.num_classes,
             num_layers=config.args.num_layers,
             num_workers=config.args.num_workers,
             path_mean=config.path_mean,
             path_std=config.path_std,
             pretrain=config.args.pretrain,
             resume_checkpoint=True,
             resume_checkpoint_path=Path('checkpoints/resnet18_e10_va0.55588.pt'),
             save_interval=config.args.save_interval,
             num_epochs=config.args.num_epochs,
             train_folder=Path('data/voc_trainval_correct/'),
             weight_decay=config.args.weight_decay)
print("+++++ Finished running 3_train.py +++++\n\n")
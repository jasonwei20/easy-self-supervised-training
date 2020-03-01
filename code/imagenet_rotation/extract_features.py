import config
from pathlib import Path
from utils_extract_features import extract_features
                
extract_features(patches_eval_folder=Path("/home/brenta/scratch/data/voc07_old/VOCdevkit/VOC2007/voc_train"),
                output_folder=Path("/home/brenta/scratch/jason/outputs/voc/example_features"),
                auto_select=False,
                batch_size=config.args.batch_size,
                checkpoints_folder=config.args.checkpoints_folder,
                classes=config.classes,
                device=config.device,
                eval_model=Path("/home/brenta/scratch/jason/checkpoints/example_models/resnet_50_in_random_hardcode_kaiming.pt"),
                num_classes=config.num_classes,
                num_layers=50,
                num_workers=config.args.num_workers,
                path_mean=config.path_mean,
                path_std=config.path_std,
                pretrain=config.args.pretrain)
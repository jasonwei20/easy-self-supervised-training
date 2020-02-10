import config
from pathlib import Path
from utils_model import extract_features
                
extract_features(patches_eval_folder=Path("/home/brenta/scratch/data/voc07/VOCdevkit/VOC2007/voc_train"),
                output_folder=Path("/home/brenta/scratch/jason/outputs/image_net/vanilla/features"),
                auto_select=False,
                batch_size=config.args.batch_size,
                checkpoints_folder=config.args.checkpoints_folder,
                classes=config.classes,
                device=config.device,
                # eval_model=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_10/resnet18_e0_mb40000_va0.80428.pt"),
                # eval_model=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_24/resnet18_e2_mb100000_va0.83949.pt"),
                # eval_model=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_25/resnet18_e0_mb10000_va0.58028.pt"),
                # eval_model=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_10/resnet18_e0_mb10000_va0.70834.pt"),
                eval_model=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_25/resnet18_e2_mb120000_va0.74405.pt"),
                num_classes=config.num_classes,
                num_layers=config.args.num_layers,
                num_workers=config.args.num_workers,
                path_mean=config.path_mean,
                path_std=config.path_std,
                pretrain=config.args.pretrain)
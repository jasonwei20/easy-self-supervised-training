import config
from pathlib import Path
from utils_model import get_predictions

# Run the ResNet on the generated patches.
print("\n\n+++++ Running 4_test.py +++++")
print("\n----- Finding validation patch predictions -----")
# Validation patches.
                
get_predictions(patches_eval_folder=Path("/home/brenta/scratch/data/imagenet_rotnet/train"),
                #patches_eval_folder=Path("/home/ifsdata/vlg/jason/easy-self-supervised-training/data/voc_trainval_full/train"),
                output_folder=Path("/home/brenta/scratch/jason/outputs/image_net/vanilla/").joinpath("resnet18_e0_mb40000_va0.80428.pt"),
                auto_select=False,
                batch_size=config.args.batch_size,
                checkpoints_folder=config.args.checkpoints_folder,
                classes=config.classes,
                device=config.device,
                eval_model=Path("/home/brenta/scratch/jason/checkpoints/image_net/vanilla/exp_10/resnet18_e0_mb40000_va0.80428.pt"),
                num_classes=config.num_classes,
                num_layers=config.args.num_layers,
                num_workers=config.args.num_workers,
                path_mean=config.path_mean,
                path_std=config.path_std,
                pretrain=config.args.pretrain)

print("----- Finished finding validation patch predictions -----\n")
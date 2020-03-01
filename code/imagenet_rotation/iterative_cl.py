import config
from pathlib import Path
from utils_model_cl import train_resnet, get_predictions
from utils_cl import *
from utils import get_log_csv_name

train_patches_eval_folder = Path("/home/brenta/scratch/data/imagenet_rotnet/train")
# train_patches_eval_folder = Path("/home/ifsdata/vlg/jason/easy-self-supervised-training/data/voc_trainval_full/train")
exp_num = 41
checkpoints_folder = Path("/home/brenta/scratch/jason/checkpoints/image_net/cl/exp_" + str(exp_num))
preds_folder = Path("/home/brenta/scratch/jason/outputs/image_net/cl/exp_" + str(exp_num))
data_folder = Path("/home/brenta/scratch/jason/data/imagenet/cl/exp_" + str(exp_num))
log_folder = Path("/home/brenta/scratch/jason/logs/imagenet/cl/exp_" + str(exp_num))
num_iterations = 100

preds_csv_name = None
train_folder_name = None
# model_name = "resnet18_e0_mb40000_va0.80428.pt"
model_name = "resnet50_e0_mb20000_va0.65573.pt"
model_path = checkpoints_folder.joinpath(model_name)
learning_rate = 0.0003
lr_decay = 0.9
num_minibatches = 0

for iteration_num in range(1, num_iterations + 1):

    preds_csv_name = f"pred_csv_{str(iteration_num)}.csv"
    preds_csv_path = preds_folder.joinpath(preds_csv_name)
    train_folder_name = "train_folder_" + str(iteration_num)
    train_folder_path = data_folder.joinpath(train_folder_name)
    print(preds_csv_path)
    print(train_folder_path)
    print(model_path)

    ######################################################################
    # Get Predictions

    get_predictions(patches_eval_folder=train_patches_eval_folder,
                    output_path=preds_csv_path,
                    model_path=model_path,
                    batch_size=128,
                    classes=config.classes,
                    device=config.device,
                    num_classes=config.num_classes,
                    num_layers=config.args.num_layers,
                    num_workers=config.args.num_workers,
                    path_mean=config.path_mean,
                    path_std=config.path_std)
    print(f"finished predictions{preds_csv_path}")

    ######################################################################
    # Find the new training set based on confidences

    def move_samples(random_samples):
        for sample in random_samples:
            source = train_patches_eval_folder.joinpath(sample)
            dest = train_folder_path.joinpath(sample)
            shutil.copyfile(source, dest)

    def get_training_list_spec(start_frac, end_frac, tuple_list, name_list, random_sample_frac, train_folder_path):
        hards = get_hards_specific(start_frac, end_frac, tuple_list)
        random_samples = get_random_sampling(random_sample_frac, hards, name_list)
        samples = hards + random_samples
        print(len(samples), "samples to move")

        for _class in ['0', '90', '180', '270']:
            train_folder_path.joinpath(_class).mkdir(parents=True, exist_ok=True)

        chunks = get_chunks(samples, 8)
        p = Pool(8)
        p.map(move_samples, chunks)

    name_list, tuple_list = training_preds_file_to_tuple_list(preds_csv_path)
    get_training_list_spec(0.0, 0.1, tuple_list, name_list, 1, train_folder_path)
    print(f"finished {train_folder_path}")
    
    ######################################################################
    # Train for a couple of epochs

    log_csv = get_log_csv_name(log_folder=log_folder)
    learning_rate = learning_rate * lr_decay
    print(learning_rate, log_csv)

    model_path, num_minibatches = train_resnet(batch_size=256,
                        checkpoints_folder=checkpoints_folder,
                        classes=config.classes,
                        color_jitter_brightness=config.args.color_jitter_brightness,
                        color_jitter_contrast=config.args.color_jitter_contrast,
                        color_jitter_hue=config.args.color_jitter_hue,
                        color_jitter_saturation=config.args.color_jitter_saturation,
                        device=config.device,
                        iteration_num=iteration_num, 
                        learning_rate=learning_rate,
                        learning_rate_decay=config.args.learning_rate_decay,
                        log_csv=log_csv,
                        num_classes=config.num_classes,
                        num_layers=config.args.num_layers,
                        num_workers=config.args.num_workers,
                        num_minibatches=num_minibatches, 
                        path_mean=config.path_mean,
                        path_std=config.path_std,
                        resume_checkpoint_path=model_path,
                        save_interval=config.args.save_interval,
                        num_epochs=1,
                        train_folder=train_folder_path,
                        weight_decay=config.args.weight_decay)

    ######################################################################
    print(f"Iteration {iteration_num} done.")
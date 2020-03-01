import operator
import random
import time
from pathlib import Path
from typing import (Dict, IO, List, Tuple)

import torchvision.models as models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import (datasets, transforms)

from utils import (get_image_paths, get_subfolder_paths)
from utils_model import create_model

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def extract_features(patches_eval_folder: Path, output_folder: Path,
                    checkpoints_folder: Path, auto_select: bool,
                    eval_model: Path, device: torch.device, classes: List[str],
                    num_classes: int, path_mean: List[float],
                    path_std: List[float], num_layers: int, pretrain: bool,
                    batch_size: int, num_workers: int) -> None:

    # Initialize the model.
    model_path = eval_model #get_best_model(checkpoints_folder=checkpoints_folder) if auto_select else eval_model

    model = create_model(num_classes=num_classes, num_layers=num_layers, pretrain=False)
    ckpt = torch.load(f=model_path)
    model.load_state_dict(  state_dict=ckpt["model_state_dict"])
    # optimizer.load_state_dict( state_dict=ckpt["optimizer_state_dict"])
    # scheduler.load_state_dict( state_dict=ckpt["scheduler_state_dict"])
                            
    # model = models.resnet18(pretrained=False)
    model = model.to(device=device)

    model.train(mode=False)
    print(f"model loaded from {model_path}")

    class ResNet50Bottom(nn.Module):
        def __init__(self, original_model):
            super(ResNet50Bottom, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-2])
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.avg_pool5 = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
            
        def forward(self, x):
            x = self.features(x)
            features_9k = self.avg_pool5(x)
            return features_9k

    res_conv_feature = ResNet50Bottom(model)

    # Confirm the output directory exists.
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the image dataset.
    dataloader = torch.utils.data.DataLoader(
        dataset=ImageFolderWithPaths(root=str(patches_eval_folder),
            transform=transforms.Compose(transforms=[
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=path_mean, std=path_std)
            ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    test_label_to_class = {0:"0", 1:"180", 2:"270", 3:"90"}
    class_num_to_class = {i: classes[i] for i in range(num_classes)}

    num_samples = len(dataloader) * batch_size
    output_features = np.zeros((num_samples, 8192))

    for batch_num, (test_inputs, test_labels, paths) in enumerate(dataloader):

        outputs = res_conv_feature(test_inputs.to(device=device))
        outputs = outputs.cpu().data.numpy()
        # print(outputs.shape)
        #(64, 2048, 2, 2)
        #(64, 8192)
        outputs = outputs.reshape(outputs.shape[0], -1)
        # print(outputs.shape)

        start_idx = batch_num * batch_size
        end_idx = start_idx + outputs.shape[0]
        output_features[start_idx:end_idx, :] = outputs
        
        for i in range(outputs.shape[0]):
            # Find coordinates and predicted class.
            image_name = "/".join(paths[i].split('/')[-2:])
    
    these_features = output_features[:2501, :]

    from numpy import save
    output_path = str(output_folder.joinpath(str(eval_model.name).split('.')[0])) + ".npy"
    print(f"features saved at {output_path}")
    save(output_path, these_features)

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

def create_model(num_layers: int, num_classes: int,
                 pretrain: bool) -> torchvision.models.resnet.ResNet:
    """
    Instantiate the ResNet model.
    Args:
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        num_classes: Number of classes in the dataset.
        pretrain: Use pretrained ResNet weights.
    Returns:
        The instantiated ResNet model with the requested parameters.
    """
    # assert num_layers in (
    #     18, 34, 50, 101, 152
    # ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {num_layers}"
    # model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
    # model = model_constructor(num_classes=num_classes)
    model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)

    for name, param in model.named_parameters():
        if name == "layer1.0.conv1.weight":
            print(param.shape)
            print(param[:3, :3, :, :])
        # if param.requires_grad:
        #     print(name, param)
    
    # print(model.layer1.0.conv1.weight.param.detach().cpu().numpy().shape)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    for name, param in model.named_parameters():
        if name == "layer1.0.conv1.weight":
            print(param.shape)
            print(param[:3, :3, :, :])


    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    # for name, param in model.named_parameters():
    #     if name == "layer1.0.conv1.weight":
    #         print(param.shape)
    #         print(param[:3, :3, :, :])
    # torch.nn.init.xavier_uniform(model.conv1.weight)
    # torch.nn.init.xavier_uniform(model.fc.weight)

    # if pretrain:
    #     pretrained = model_constructor(pretrained=True).state_dict()
    #     if num_classes != pretrained["fc.weight"].size(0):
    #         del pretrained["fc.weight"], pretrained["fc.bias"]
    #     model.load_state_dict(state_dict=pretrained, strict=False)
    return model

if __name__ == "__main__":

    model = create_model(num_classes=4,
                         num_layers=50,
                         pretrain=False)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=0.0001,
                           weight_decay=0.0001)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,
                                           gamma=0.8)

    # model = model.to(device=device)
    torch.save(obj={
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, f=str("/home/brenta/scratch/jason/checkpoints/example_models/resnet_50_in_random_hardcode_kaiming.pt"))

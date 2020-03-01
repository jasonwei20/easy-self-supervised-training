
import numpy as np
from numpy import save
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

model_path = Path("/home/brenta/scratch/jason/checkpoints/example_models/resnet50_in_pretrained.pt")

if __name__ == "__main__":

    model = torchvision.models.resnet50(pretrained=False, num_classes=4)
    ckpt = torch.load(f=model_path)
    model.load_state_dict(state_dict=ckpt["model_state_dict"])

    for name, param in model.named_parameters():
        if name == "layer1.0.conv1.weight":
            print(param.shape)
            print(param[:3, :3, :, :])
        if name == "layer2.0.conv1.weight":
            print(param.shape)
            print(param[:3, :3, :, :])

import torch
import torchvision
import torch.nn as nn

if __name__ == "__main__":

    # model = torchvision.models.resnet50(pretrained=True)
    model = torchvision.models.resnet50(pretrained=False, num_classes=4)

    for name, param in model.named_parameters():
        if name == "layer1.0.conv1.weight":
            print(param.shape)
            print(param[:3, :3, :, :])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    for name, param in model.named_parameters():
        if name == "layer1.0.conv1.weight":
            print(param.shape)
            print(param[:3, :3, :, :])

    torch.save(obj={
        "model_state_dict": model.state_dict(),
    }, f=str("/home/brenta/scratch/jason/checkpoints/example_models/resnet_50_in_random_uniform.pt"))
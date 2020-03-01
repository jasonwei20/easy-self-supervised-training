
import numpy as np
from numpy import save
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

patches_eval_folder = Path("/home/brenta/scratch/data/voc07_old/VOCdevkit/VOC2007/voc_train")
model_path = Path("/home/brenta/scratch/jason/checkpoints/example_models/resnet_50_in_random_clean.pt")
output_folder = Path("/home/brenta/scratch/jason/outputs/voc/example_features/conv4")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_classes = 4
num_workers = 8
path_mean, path_std = ( [0.40853017568588257, 0.4573926329612732, 0.48035722970962524], 
                        [0.28722450137138367, 0.27334490418434143, 0.2799932360649109])

class ResNet50BottomConv4(nn.Module):
    def __init__(self, original_model):
        super(ResNet50BottomConv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.avg_pool5 = nn.AvgPool2d(kernel_size=8, stride=3, padding=0)
        
    def forward(self, x):
        x = self.features(x)
        features_9k = self.avg_pool5(x)
        return features_9k

if __name__ == "__main__":

    model = torchvision.models.resnet50(pretrained=False, num_classes=4)
    ckpt = torch.load(f=model_path)
    model.load_state_dict(state_dict=ckpt["model_state_dict"])
    model = model.to(device=device)
    model.train(mode=False)
    print(f"model loaded from {model_path}")

    res_conv_feature = ResNet50BottomConv4(model)

    output_folder.mkdir(parents=True, exist_ok=True) # Confirm the output directory exists.

    # Load the image dataset.
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(root=str(patches_eval_folder),
            transform=transforms.Compose(transforms=[
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=path_mean, std=path_std)
            ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    num_samples = len(dataloader) * batch_size
    output_features = np.zeros((num_samples, 9216))

    for batch_num, (test_inputs, test_labels) in enumerate(dataloader):

        outputs = res_conv_feature(test_inputs.to(device=device))
        outputs = outputs.cpu().data.numpy()
        outputs = outputs.reshape(outputs.shape[0], -1) #reshape from (64, 1024, 14, 14) to (64, 9216)

        start_idx = batch_num * batch_size
        end_idx = start_idx + outputs.shape[0]
        output_features[start_idx:end_idx, :] = outputs
    
    these_features = output_features[:2501, :]

    output_path = str(output_folder.joinpath(str(model_path.name).split('.')[0])) + ".npy"
    save(output_path, these_features)
    print(f"features saved at {output_path}")

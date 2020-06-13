"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

# Importing libraries
import os
import torch
import torch.nn as nn

# From https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
models_dir = os.path.expanduser('~/.torch/models')
model_name = 'alexnet-owt-4df8aa71.pth'

# Class containing definition for AlexNet
class AlexNet(nn.Module):
    def __init__(self, classes=1000):
        super(AlexNet, self).__init__()

        # See the model architecture from the README
        self.features = nn.Sequential(
            # First Layer Group, input dims --> 224x224x3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # (224-11)/4 + 1 = 55 | Dims now --> 55x55x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (55-3)/2 + 1 = 27 | Dims now --> 27x27x64

            # Second Layer Group, input dims --> 27x27x64
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # Same Convolution | Dims now --> 27x27x192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2 + 1 | Dims now --> 13x13x192

            # Third Layer, input dims --> 13x13x192
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # Same Convolution | Dims now --> 13x13x384
            nn.ReLU(inplace=True),

            # Fourth Layer, input dims --> 13x13x384
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # Same Convolution | Dims now --> 13x13x256
            nn.ReLU(inplace=True),

            # Fifth Layer, input dims --> 13x13x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Same Convolution | Dimns now --> 13x13x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # (13-3)/2 + 1 | Dims now --> 6x6x256
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            # First Fully Connected Layer | input dims --> 6x6x256
            nn.Linear(256*6*6, 4096), # Unfolding volume into linear layer | Dims now --> 1x4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # Second Fully Connected Layer | input dims --> 1x4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Third Fully Connected Layer | input dims --> 1x4096
            nn.Linear(4096, classes)
        )

    def forward(self, x):
        """
        One complete forward iteration.
        """
        x = self.features()
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

def alexnet(pretrained=True, **kwargs):
    """
    Define the AlexNet model.
    params
    pretrained : Boolean , default True | get pre-trained weights from pytorch
    """
    model = AlexNet(**kwargs)
    if pretrained:model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model

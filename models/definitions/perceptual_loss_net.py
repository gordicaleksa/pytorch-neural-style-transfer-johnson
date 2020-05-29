"""
    VGG16 deep learning model is used as the perceptual loss (network).
    More detail about the VGG architecture (if you want to understand magic/hardcoded numbers) can be found here:

    https://github.com/pytorch/vision/blob/3c254fb7af5f8af252c24e89949c54a3461ff0be/torchvision/models/vgg.py
"""

from collections import namedtuple
import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        # Keeping eval() mode only for consistency - it only affects BatchNorm and Dropout both of which we won't use
        vgg16 = models.vgg16(pretrained=True, progress=show_progress).eval()
        vgg_pretrained_features = vgg16.features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


# Set the perceptual loss network to be VGG16
PerceptualLossNet = Vgg16

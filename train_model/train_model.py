import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

data_dir  = 'data/garbage_lassification'

classes = os.listdir(data_dir)
print(classes)

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)

dataset

import matplotlib.pyplot as plt

def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))  # Assuming img is a tensor in (C, H, W) format
    plt.axis('off')  # Optional: Hide axis
    plt.show()  # Add this line to display the image


img, label = dataset[7]
show_sample(img, label)

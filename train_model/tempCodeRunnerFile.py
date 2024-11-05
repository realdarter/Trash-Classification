import os
import time
import json
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
import random
import os
import yaml


# Load training configuration

with open("configs/training.yml", 'r') as stream:
    try:
        train_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
print(train_config)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = train_config['image_size']
# Define transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.44, 0.46, 0.35), (0.01, 0.01, 0.02))
])

# Download or prepare your custom dataset and create DataLoader
root_dir = train_config['raw_data_dir']
whole_data = datasets.ImageFolder(root=root_dir, transform=transform)
# Data Labelling
labels = whole_data.targets
labels
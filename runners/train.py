import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
import random
import os
import yaml
import numpy as np 
import sys

current_script_directory = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory
parent_directory = os.path.abspath(os.path.join(current_script_directory, os.pardir))
sys.path.append(parent_directory)

from flora_utils.flora_utils.src.preprocessing import get_data_loaders

# To reprocuding the data splitting

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

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
dataset_path = train_config['raw_data_dir']

# Split the data:
train_ratio = train_config['train_ratio']
flip_p = train_config['flip_p']
batch_size = train_config['batch_size']

(train_loader, train_data_len) = get_data_loaders(data_dir=dataset_path, batch_size=batch_size, train=True, flip_p=flip_p, image_size=image_size, train_ratio=train_ratio)

(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(data_dir=dataset_path, batch_size=batch_size, train=False, flip_p=flip_p, image_size=image_size, train_ratio=train_ratio)

# classes = get_classes(dataset_path)

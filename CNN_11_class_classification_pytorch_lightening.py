import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import pytorch_lightning as pl
# Define the folder where your images are stored
os.chdir('/scratch1/ssoltani/workshop/11_FloraMask/')

class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, test_size=0.2, val_size=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        # Load all image files and labels
        file_paths, labels = [], []
        for class_id, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_paths.append(os.path.join(class_dir, file_name))
                    labels.append(class_id)

        # Split data into train, val, and test sets
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            file_paths, labels, test_size=self.test_size, stratify=labels)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=self.val_size / (1 - self.test_size), stratify=train_labels)

        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomImageDataset(train_paths, train_labels, transform=self.transform)
            self.val_dataset = CustomImageDataset(val_paths, val_labels, transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_dataset = CustomImageDataset(test_paths, test_labels, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Usage

# Use the data module
data_module = DataModule('dataset/01_myDiv_tree_spec_training_photos/')
print(data_module.batch_size)
print(data_module.data_dir)
data_module.setup()
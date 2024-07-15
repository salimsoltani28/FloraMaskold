# This file builds dataloader for the dataset
import random
import torch
import warnings
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .dataset import SegmentationDataset
from torch.utils.data import DataLoader
import albumentations as A
from transformers import MaskFormerImageProcessor

def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    warnings.warn(repr(exn))
    return True

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_preprocessor():
    # Create a preprocessor
    return MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

def collate_fn(batch, preprocessor=get_preprocessor()):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    return batch

def build_img_transform():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the desired size
        transforms.ToTensor()           # Convert to PyTorch tensor
    ])
    return transform

def build_dataset(config):
    img_transform = build_img_transform()
    # Load dataset
    dataset = SegmentationDataset(config['data']['image_dir'], config['data']['mask_dir'], transform=img_transform)
    print('dataset len: ', len(dataset))
    return dataset

def build_loader(config):
    dataset = build_dataset(config=config)
    print('successfully built dataset')

    # Split dataset into training and validation sets
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    
    # Create Subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    # DataLoaders
    data_loader_train = DataLoader(
        train_subset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],  # Adjusted from config
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    data_loader_val = DataLoader(
        val_subset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],  # Adjusted from config
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    return train_subset, val_subset, data_loader_train, data_loader_val

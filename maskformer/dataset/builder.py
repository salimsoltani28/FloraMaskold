#THid file builds dataloader for the dataset
import random
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

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    
    return batch

def build_img_transform():
    # Define transformations
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    train_transform = A.Compose([
        A.LongestMaxSize(max_size=1333),
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])

    test_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),

    ])
    
    return train_transform, test_transform

def build_dataset(config):
    img_transform, _ = build_img_transform()
    # Load dataset
    dataset = SegmentationDataset(config.image_dir, config.mask_dir, transform=img_transform)
    print('dataset len: ', len(dataset))
    return dataset

def build_loader(config):
    
    #local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    dataset_train = build_dataset(config=config)
    print('successfully build train dataset')
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn 
    )
    return dataset_train, data_loader_train
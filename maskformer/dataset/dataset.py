import os
from PIL import Image
from torch.utils.data import Dataset
import torch
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        #TODO: Don't hardcode
        self.idx_to_class = {
            "0": "Acer pseudoplatanus",
            "1": "Aesculus hippocastanum",
            "2": "Betula pendula",
            "3": "Carpinus betulus",
            "4": "Fagus sylvatica",
            "5": "Fraxinus excelsior",
            "6": "Prunus avium",
            "7": "Quercus petraea",
            "8": "Sorbus aucuparia",
            "9": "Tilia platyphyllos",
            "11": "Grass"
            
        }
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)

    def create_binary_masks(self, mask):
        unique_classes = torch.unique(mask)
        N = len(unique_classes)
        h, w = mask.shape
        binary_masks = torch.zeros((N, h, w), dtype=torch.uint8)

        for i, cls in enumerate(unique_classes):
            binary_masks[i] = (mask == cls).to(torch.uint8)
        
        return binary_masks, unique_classes
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        #for mask name append mask_ in front of image name
        mask_name = 'mask_'+img_name
        #make it a png
        mask_name = mask_name.replace('.jpg', '.png')
        
        print(f'idx: {idx} img name :{img_name} mask_name :{mask_name}')
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            ori_image = Image.open(img_path).convert("RGB")
            ori_mask = Image.open(mask_path).convert("L")  # Assuming mask is in grayscale
        except Exception as e:
            #TODO: To be removed after fixing the dataset
            print(f"Error loading image: {img_path}: {e}")
            # Load a blank image
            ori_image = Image.new("RGB", (256, 256))
            #Load a blank mask
            ori_mask = Image.new("L", (256, 256))
            
        if self.transform:
            image = self.transform(ori_image)
            mask = self.transform(ori_mask).squeeze(0) # Remove the channel dimension
            
        #create binary masks
        binary_mask, class_labels = self.create_binary_masks(mask)
        
        #get unique values in the mask as class labels
        class_labels = torch.Tensor(class_labels)
        binary_mask = torch.Tensor(binary_mask)
        
        
        
        
        return image, mask, class_labels, binary_mask

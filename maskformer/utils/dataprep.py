import os
import shutil

def collapse_subfolders(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                shutil.move(file_path, target_dir)
            except:
                continue

# Define source and target directories
source_image_dir  = '/mnt/gsdata/projects/panops/Labeled_data_seprated_in_Folder/image'
source_mask_dir = '/mnt/gsdata/projects/panops/Labeled_data_seprated_in_Folder/image_mask'
target_image_dir = '/mnt/gsdata/projects/panops/Labeled_data_seprated_in_Folder/image_all'
target_mask_dir = '/mnt/gsdata/projects/panops/Labeled_data_seprated_in_Folder/image_mask_all'

# Collapse subfolders for images
collapse_subfolders(source_image_dir, target_image_dir)

# Collapse subfolders for masks
collapse_subfolders(source_mask_dir, target_mask_dir)

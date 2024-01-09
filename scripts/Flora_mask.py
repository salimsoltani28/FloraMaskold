
# Now import PyTorch, NumPy, etc.
import torch
import numpy as np
# ... other imports

import os
import sys
import subprocess
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
#!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
import torch
print(torch.cuda.is_available())

#$env:KMP_DUPLICATE_LIB_OK="TRUE"

#Import the images
image = cv2.imread('data/1_example_photos_iNat/Acer pseudoplatanus0000018.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
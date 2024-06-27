import torch
import torch.nn as nn
from .builder import MODELS
from utils import get_logger, load_class_from_config
from typing import Optional
import torch
from transformers import MaskFormerForInstanceSegmentation

@MODELS.register_module()
class MaskFormer2(nn.Module):
    def __init__(self):
        super(MaskFormer2, self).__init__()
        self.logger = get_logger()
        # Replace the head of the pre-trained model
        id2label = {
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
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)

    def forward(self, batch):
        outputs = self.model(batch["pixel_values"],
                class_labels=batch["class_labels"],
                mask_labels=batch["mask_labels"])
        return outputs

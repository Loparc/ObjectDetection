from typing import Any

from converter import read_content
DIR_TO_XML_FILE = "/dtu/datasets1/02516/potholes/annotations"
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
folder = Path(DIR_TO_XML_FILE)

        
'''
This dataloader outputs 1 image and his boxes.

'''
class PotHoleDataLoader(Dataset):
    
    def __init__(self, annotations_dir : Path = Path("/dtu/datasets1/02516/potholes/annotations"), 
                 images_dir : Path = Path("/dtu/datasets1/02516/potholes/images"),
                 transformations = None) -> None:
        self.dataset = []
        self.transf = transformations
        for file_path, image_path in zip(annotations_dir.iterdir(),images_dir.iterdir()):
            
            if file_path.is_file() and image_path.is_file():
                content = file_path.read_text(encoding="utf-8", errors="replace")
                
                _,boxes = read_content(content)
                self.dataset.append((np.array(boxes, dtype=np.float32), image_path))
    
    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, index, state="training") -> Any:
        
        (boxes, image_path) = self.dataset[index]
        
        
        img = Image.open(image_path).convert("RGB")
        if state == "training":
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = np.array(boxes, dtype = np.float32)
            
        if self.transf:
            img = self.transf(img)
        
        return img, boxes


    
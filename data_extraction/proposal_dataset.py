import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = Path("/dtu/datasets1/02516/potholes/images")
PROPOSALS_JSON = ROOT / "object_proposals" / "proposals_labeled.json"
SPLITS_JSON = ROOT / "data_extraction" / "splits.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def default_transform():
    """to add augmentation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def fixed_val_subset(labels, num_samples, seed=42):
    # same subset every time, keeps the real proportion of potholes
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    num_pos = round(num_samples * len(pos) / len(labels))
    idx = np.concatenate([rng.choice(pos, num_pos, replace=False),
                          rng.choice(neg, num_samples - num_pos, replace=False)])
    return np.sort(idx)

""" Dataset of object proposals for one split (train/val/test)
All images of the split are loaded into memory and all their proposals
are stored as flat numpy arrays: for each proposal, the index of its image,
its box (xmin, ymin, xmax, ymax) and its label (0 = background, 1 = pothole)"""

class ProposalDataset(Dataset):
    """Each sample = (224x224 crop of one proposal, label 0/1)."""

    def __init__(self, split, proposals=None, transform=None):
        """
        :param split: "train", "val" or "test"
        :param proposals: dictionary already loaded from the JSON (avoids reading it multiple times)
        :param transform: transformations applied to each crop
        """
        if proposals is None:
            proposals = json.load(open(PROPOSALS_JSON))
        self.image_names = json.load(open(SPLITS_JSON))[split]
        self.transform = transform or default_transform()

        # Load all images of this split into memory (only once)
        paths = {p.stem: p for p in IMAGES_DIR.iterdir()}
        self.images = [Image.open(paths[n]).convert("RGB") for n in self.image_names]

        # Store all proposals as flat numpy arrays (much lighter than Python lists)
        img_idx, boxes, labels = [], [], []
        for i, name in enumerate(self.image_names):
            arr = np.asarray(proposals[name], dtype=np.int32)   # (M, 5)
            img_idx.append(np.full(len(arr), i, dtype=np.int32))
            boxes.append(arr[:, :4])
            labels.append(arr[:, 4])
        self.img_idx = np.concatenate(img_idx)                   # which image each proposal belongs to
        self.boxes = np.concatenate(boxes)                       # xmin, ymin, xmax, ymax
        self.labels = np.concatenate(labels).astype(np.int64)    # 0 background, 1 pothole

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[self.img_idx[i]]
        W, H = img.size
        x1, y1, x2, y2 = self.boxes[i]

        # Make sure the box lies inside the image and is at least 1 pixel wide/high
        x1, y1 = min(max(0, x1), W - 1), min(max(0, y1), H - 1)
        x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))

        crop = img.crop((x1, y1, x2, y2))
        return self.transform(crop), self.labels[i]
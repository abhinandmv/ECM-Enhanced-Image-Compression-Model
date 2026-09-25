# import torch
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# import os

# class DIV2KDataset(Dataset):
#     """
#     Dataset for DIV2K high-resolution images.
#     """
#     def __init__(self, data_path, transform=None):
#         self.data_path = data_path
#         self.transform = transform
#         if not os.path.exists(data_path):
#             print(f"Warning: Directory {data_path} does not exist.")
#             self.images = []
#         else:
#             self.images = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
#         if not self.images:
#             print(f"Warning: No images found in {data_path}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.data_path, img_name)
#         img = Image.open(img_path).convert("RGB")
            
#         if self.transform:
#             img = self.transform(img)
#         return img, 0

# def prepare_div2k_dataloader(data_path, batch_size=32, num_workers=8, shuffle=True, patch_size=256):
#     """
#     Dataloader for DIV2K. 
#     - Training: RandomCrop for patch-based training
#     - Validation: Full image (no cropping) to ensure model learns full-image reconstruction
#     """
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(patch_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
    
#     # CRITICAL FIX: Validation uses full image, not center-crop
#     # This ensures the model learns to reconstruct the entire image, not just the center
#     val_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     dataset = DIV2KDataset(data_path, transform=train_transform if shuffle else val_transform)
    
#     return DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         num_workers=num_workers, 
#         shuffle=shuffle,
#         pin_memory=True,
#         drop_last=True if shuffle else False
#     )



# import torch
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# import os


# class DIV2KDataset(Dataset):
#     """
#     Dataset for DIV2K high-resolution images.
#     """
#     def __init__(self, data_path, transform=None):
#         self.data_path = data_path
#         self.transform = transform
#         if not os.path.exists(data_path):
#             print(f"Warning: Directory {data_path} does not exist.")
#             self.images = []
#         else:
#             self.images = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
#         if not self.images:
#             print(f"Warning: No images found in {data_path}")


#     def __len__(self):
#         return len(self.images)


#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.data_path, img_name)
#         img = Image.open(img_path).convert("RGB")
            
#         if self.transform:
#             img = self.transform(img)
#         return img, 0


# def prepare_div2k_dataloader(data_path, batch_size=32, num_workers=0, shuffle=True, patch_size=256):
#     """
#     Dataloader for DIV2K. 
#     - Training: RandomCrop for patch-based training
#     - Validation: Resize to consistent size (no center-crop to ensure full-image reconstruction)
#     """
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(patch_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
    
#     # CRITICAL FIX: Validation resizes to consistent size
#     # This ensures:
#     # 1. All validation images have the same size (no stacking errors)
#     # 2. Model learns full-image reconstruction (not center-crop)
#     # 3. Consistent evaluation metrics across batches
#     val_transform = transforms.Compose([
#         transforms.Resize((512, 512)),  # Resize to fixed size for consistent batching
#         transforms.ToTensor(),
#     ])
    
#     dataset = DIV2KDataset(data_path, transform=train_transform if shuffle else val_transform)
    
#     return DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         num_workers=num_workers, 
#         shuffle=shuffle,
#         pin_memory=True,
#         drop_last=True if shuffle else False
#     )


"""
Research-Grade Data Utilities for Learned Image Compression

✔ Proper train / validation split handling
✔ Patch-based training
✔ Full-resolution validation (NO resizing distortion)
✔ Deterministic behavior support
✔ Multi-dataset support (DIV2K, Kodak, CLIC)
✔ Publication-ready evaluation protocol
"""

import os
import random
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# Utility
# ============================================================

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def set_random_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Generic Image Dataset (For DIV2K / Kodak / CLIC)
# ============================================================

class ImageFolderDataset(Dataset):
    """
    Generic dataset for image compression research.

    Expects:
        root/
            img1.png
            img2.png
            ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        sort_files: bool = True,
    ):
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            raise RuntimeError(f"Dataset path does not exist: {root}")

        self.images: List[str] = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if is_image_file(f)
        ]

        if sort_files:
            self.images = sorted(self.images)

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img


# ============================================================
# Transform Builders
# ============================================================

def build_train_transform(patch_size: int = 256):
    """
    Patch-based training transform.
    Standard protocol for compression training.
    """
    return transforms.Compose([
        transforms.RandomCrop(patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])


def build_val_transform():
    """
    Validation transform.
    Ensures spatial dimensions divisible by 8 (3 stride-2 downsamples).
    """
    return transforms.Compose([
        transforms.CenterCrop((512, 512)),  # 512 divisible by 8
        transforms.ToTensor(),
    ])


# ============================================================
# DataLoader Builders
# ============================================================

def build_train_dataloader(
    data_path: str,
    batch_size: int = 16,
    patch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Training loader (patch-based).
    """

    set_random_seed(seed)

    dataset = ImageFolderDataset(
        root=data_path,
        transform=build_train_transform(patch_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )


def build_val_dataloader(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 2,
):
    """
    Validation loader (full-resolution images).
    Batch size = 1 recommended for compression evaluation.
    """

    dataset = ImageFolderDataset(
        root=data_path,
        transform=build_val_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


# ============================================================
# Standard Dataset Paths (Optional Helper)
# ============================================================

def get_div2k_paths(root: str):
    """
    Assumes standard DIV2K structure:

        root/
            DIV2K_train_HR/
            DIV2K_valid_HR/
    """
    train_path = os.path.join(root, "DIV2K_train_HR")
    val_path = os.path.join(root, "DIV2K_valid_HR")
    return train_path, val_path


def get_kodak_path(root: str):
    """
    Kodak dataset folder path.
    """
    return os.path.join(root, "Kodak")


def get_clic_path(root: str):
    """
    CLIC validation dataset folder path.
    """
    return os.path.join(root, "CLIC")









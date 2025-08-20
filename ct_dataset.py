import os
import cv2
import numpy as np
from PIL import Image
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.signal import wiener
from skimage.metrics import structural_similarity as ssim



def apply_clahe(image):
    """Apply CLAHE to improve local contrast before filtering."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    return cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)



# Improved Wiener filter
def improved_wiener_filter(img_np):
    """Apply Improved Wiener Filter with SSIM-based optimization."""
    filtered = wiener(img_np, mysize=5)

    # Handle small images
    if img_np.shape[0] < 7 or img_np.shape[1] < 7:
        return filtered
    
    rng = float(img_np.max()) - float(img_np.min())
    if rng < 1e-6:
        return filtered  # nothing to gain from SSIM blend if image nearly constant

    # Simple SSIM-based blend (retain some original structure)
    ssim_index = ssim(img_np, filtered, channel_axis=-1, data_range=rng)
    alpha = min(max(ssim_index, 0.2), 0.8)
    blended = alpha * filtered + (1 - alpha) * img_np
    return blended

class CTImageSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, image_size=(256, 256), augment=False, use_clahe=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe


        # only image files
        self.images = [f for f in sorted(os.listdir(self.image_dir))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # if training with masks: keep only images that actually have a mask file
        if mask_dir:
            case_name = os.path.basename(os.path.normpath(self.image_dir))  # e.g. "Test cases"
            def _mask_exists(stem: str) -> bool:
                p1 = os.path.join(self.mask_dir, f"{stem}.npy")                 # flat: data/masks_npy/<stem>.npy
                p2 = os.path.join(self.mask_dir, case_name, f"{stem}.npy")      # per-case: data/masks_npy/<case>/<stem>.npy
                return os.path.exists(p1) or os.path.exists(p2)

            kept = [f for f in self.images if _mask_exists(Path(f).stem)]
            print(f"[CTDataset] Using {len(kept)} pairs out of {len(self.images)} images.")
            self.images = kept

        self.masks = None  # not used; avoid confusion

        self.to_tensor = transforms.ToTensor()
        
        

    def __len__(self):
        return len(self.images)
    
    

    def _apply_augmentations(self, img_np, mask_np):
        """Apply synchronized augmentations to image and mask."""

        # Random horizontal flip
        if random.random() > 0.5:
            img_np = np.fliplr(img_np).copy()
            mask_np = np.fliplr(mask_np).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img_np = np.flipud(img_np).copy()
            mask_np = np.flipud(mask_np).copy()

        # Random rotation (±10°)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR)
            mask_np = cv2.warpAffine(mask_np, M, (w, h), flags=cv2.INTER_NEAREST)

        # Small random crop & resize back
        if random.random() > 0.5:
            h, w = img_np.shape[:2]
            crop_x = random.randint(0, int(w * 0.1))
            crop_y = random.randint(0, int(h * 0.1))
            end_x = w - crop_x
            end_y = h - crop_y
            img_np = img_np[crop_y:end_y, crop_x:end_x]
            mask_np = mask_np[crop_y:end_y, crop_x:end_x]
            img_np = cv2.resize(img_np, self.image_size, interpolation=cv2.INTER_LINEAR)
            mask_np = cv2.resize(mask_np, self.image_size, interpolation=cv2.INTER_NEAREST)

        return img_np, mask_np
    
    

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img_name = os.path.splitext(self.images[idx])[0]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)

        # Convert grayscale to RGB if needed
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        # ✅ Resize first (ensures consistent preprocessing before Wiener filter)
        img_np = cv2.resize(img_np, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # ✅ Optional CLAHE
        if self.use_clahe:  
            img_np = apply_clahe(img_np)

        # ✅ Apply Wiener filter
        img_np = improved_wiener_filter(img_np)

        # ✅ Handle NaNs and out-of-range values
        img_np = np.nan_to_num(img_np, nan=0.0, posinf=255.0, neginf=0.0)
        img_np = np.clip(img_np, 0, 255)

        # Load mask from .npy file (if we are in supervised mode)
        if self.mask_dir:
            case_name = os.path.basename(os.path.normpath(self.image_dir))  # e.g. "Test cases"
            stem = img_name
            p1 = os.path.join(self.mask_dir, f"{stem}.npy")
            p2 = os.path.join(self.mask_dir, case_name, f"{stem}.npy")
            mask_path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

            if mask_path is None:
                mask = np.zeros(self.image_size, dtype=np.float32)
            else:
                m = np.load(mask_path)
                m = m[0] if m.ndim == 3 else m         # Load and ensure shape [H, W]

                # ✅ Resize mask with NEAREST to avoid interpolation artifacts
                if m.shape != self.image_size:
                    m = cv2.resize(m, self.image_size, interpolation=cv2.INTER_NEAREST)

                # Normalize mask to [0,1]
                mask = (m > 0).astype(np.float32)

                # Warn if empty mask
                if np.sum(mask) == 0:
                    print(f"[WARN] Empty mask for {img_name}")
        else:
            mask = np.zeros(self.image_size, dtype=np.float32)

        # ✅ Apply synchronized augmentations if enabled and training
        if self.train and self.augment:
            img_np, mask = self._apply_augmentations(img_np, mask)

        # Convert to tensors
        image_tensor = self.to_tensor(img_np.astype(np.uint8))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return image_tensor, mask_tensor


import sys
log_file = open("generate_masks_log.txt", "w")
sys.stdout = sys.stderr = log_file



import os
from datetime import datetime
import cv2
import json 
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from models.resnet_segmentor import ResNetSegmentationModel
from models.ct_dataset import CTImageSegmentationDataset, improved_wiener_filter
from dice_bce_loss import DiceBCELoss

# === Added for feature extraction ===
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from skimage.filters import gaussian
import math
from embedding_utils import ResNetEncoder, extract_feature_vector

# Initialize deep feature extractor once
deep_encoder = ResNetEncoder().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()






# Configuration
INPUT_DIR = Path("data/raw/Test cases")
ANNOTATION_DIR = Path("data/masks_npy")
OUTPUT_PNG_DIR = Path("data/masks_png")
CHECKPOINT_PATH = Path("checkpoints/test_cases_segmentor.pth")
RESIZE_SHAPE = (256, 256)

# Ensure output directories exist
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Define green color range (in HSV)
LOWER_GREEN = np.array([30, 20, 20])
UPPER_GREEN = np.array([95, 255, 255])


def extract_green_box_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Save mask for debug
    os.makedirs("debug_masks", exist_ok=True)
    cv2.imwrite(f"debug_masks/debug_green_mask.png", mask)

    return contours, mask

def mask_to_polygon_annotation(contours, image_shape, image_filename):
    shapes = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.squeeze()
        if contour.ndim != 2:
            continue
        shape = {
            "label": "disease",
            "points": contour.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1]
    }




def generate_masks():
    print("[INFO] Generating masks from embedded green boxes...")   
    image_paths = sorted([*INPUT_DIR.glob("*.png"),
                      *INPUT_DIR.glob("*.jpg"),
                      *INPUT_DIR.glob("*.jpeg")])


    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not load: {img_path.name}")
            continue

        # === Apply Improved Wiener Filter + NaN sanitization (match ct_dataset.py) ===
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB for filtering
        img = cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_LINEAR)  # match training size
        img = improved_wiener_filter(img)
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # back to BGR for OpenCV ops


        contours, green_mask = extract_green_box_mask(img)
        if not contours:
            print(f"[SKIP] No green box found in: {img_path.name}")
            continue

        # Save debug green mask view
        os.makedirs("debug_masks", exist_ok=True)
        cv2.imwrite(f"debug_masks/{img_path.stem}_green_mask.png", green_mask)

        # Generate binary mask from contour for .npy output
        solid_mask = np.zeros(green_mask.shape, dtype=np.uint8)
        cv2.drawContours(solid_mask, contours, -1, 255, -1)
        
        # ✅ Resize both image and mask together before saving
        img = cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(solid_mask, RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)
        normalized_mask = (mask_resized > 0).astype(np.float32)

        if np.sum(normalized_mask) == 0:
            print(f"[WARN] Empty mask for {img_path.name}")

        
        # === FEATURE EXTRACTION BLOCK ===
        features_dir = Path("data/features/Test cases")
        features_dir.mkdir(parents=True, exist_ok=True)

        # Crop lesion region from original image
        mask_bool = solid_mask > 0
        if mask_bool.any():
            x, y, w, h = cv2.boundingRect(mask_bool.astype(np.uint8))
            cropped_img  = img[y:y+h, x:x+w]
            cropped_mask = solid_mask[y:y+h, x:x+w]
        else:
            cropped_img  = img
            cropped_mask = solid_mask

        # --- 1. MLDN Features ---
        # Apply Kirsch masks (8 directions)
        kirsch_kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
        ]
        mldn_feats = []
        gray_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        for k in kirsch_kernels:
            mldn_feats.append(cv2.filter2D(gray_crop, -1, k).mean())

        # --- 2. MRELBP + Entropy ---
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_crop, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points+3), range=(0, n_points+2))
        lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
        entropy_val = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-6))

        # --- 3. Shape features ---
        props = regionprops(cropped_mask.astype(np.uint8))[0]
        shape_feats = [
            props.area,
            props.perimeter,
            props.eccentricity,
            props.solidity
        ]

        # --- 4. Deep features ---
        deep_feats = extract_feature_vector(str(img_path), deep_encoder).numpy()

        # Combine all features
        feature_vector = np.concatenate([
            np.array(mldn_feats),
            lbp_hist,
            np.array([entropy_val]),
            np.array(shape_feats),
            deep_feats
        ])

        # Save features
        np.savez(features_dir / f"{img_path.stem}.npz", features=feature_vector)
        print(f"[OK] Saved features: {features_dir / (img_path.stem + '.npz')}")

        subfolder = img_path.parent.name  # e.g., 'Test cases'
        
        # --- NPY under per-case folder ---
        npy_path = ANNOTATION_DIR / subfolder / (img_path.stem + ".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, normalized_mask[np.newaxis, ...])

        # --- PNG under per-case folder (you already had this, keep it) ---
        png_path = OUTPUT_PNG_DIR / subfolder / (img_path.stem + ".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((normalized_mask * 255).astype(np.uint8)).save(png_path)

        # Save corresponding JSON annotation from original (not resized) contour
        json_data = mask_to_polygon_annotation(contours, img.shape[:2], img_path.name)
        json_path = Path("data/annotations") / subfolder / f"{img_path.stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
            
        print(f"[OK] Saved: {npy_path.name} | {png_path.name} | {json_path.name}")




def train_model():
    print("\n[INFO] Starting model training...")
    
    losses = []
    N = 5  # save visualizations every N epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_root = ANNOTATION_DIR / INPUT_DIR.name   # e.g. data/masks_npy/Test cases
    dataset = CTImageSegmentationDataset(image_dir=str(INPUT_DIR),
                                         mask_dir=str(mask_root),
                                         train=True, augment=True, use_clahe=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    model = ResNetSegmentationModel(debug_shapes=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    for epoch in range(10):  # <-- put your desired number of epochs here
        model.train()
        running_loss = 0.0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # === Save visualizations every N epochs ===
        if epoch % N == 0:
            model.eval()
            with torch.no_grad():
                sample_images, sample_masks = next(iter(dataloader))
                sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)
                sample_outputs = model(sample_images)

                for i in range(min(len(sample_images), 3)):
                    pred_mask = torch.sigmoid(sample_outputs[i]).cpu().numpy()[0]
                    gt_mask = sample_masks[i].cpu().numpy()[0]
                    img = sample_images[i].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC

                    # Normalize image for saving
                    img = (img * 255).astype(np.uint8)
                    pred_mask_img = (pred_mask > 0.5).astype(np.uint8) * 255
                    gt_mask_img = (gt_mask > 0.5).astype(np.uint8) * 255

                    # Create overlay for GT mask
                    overlay_gt = img.copy()
                    overlay_gt[gt_mask_img > 0] = [255, 0, 0]  # red overlay
                    overlay_gt = cv2.addWeighted(img, 0.7, overlay_gt, 0.3, 0)

                    # Overlay for predicted mask
                    overlay_pred = img.copy()
                    overlay_pred[pred_mask_img > 0] = [0, 255, 0]  # green overlay
                    overlay_pred = cv2.addWeighted(img, 0.7, overlay_pred, 0.3, 0)
                    
                    # Ensure previews match model input size
                    target_size = RESIZE_SHAPE if 'RESIZE_SHAPE' in globals() else (256, 256)

                    img_vis         = cv2.resize(img,         target_size, interpolation=cv2.INTER_LINEAR)
                    overlay_gt_vis  = cv2.resize(overlay_gt,  target_size, interpolation=cv2.INTER_LINEAR)
                    overlay_pred_vis= cv2.resize(overlay_pred,target_size, interpolation=cv2.INTER_LINEAR)

                    combined = np.hstack([
                        img_vis,         # original
                        overlay_gt_vis,  # GT overlay
                        overlay_pred_vis # Pred overlay
                    ])
                    
                    # Save combined image
                    os.makedirs("data/training_visuals", exist_ok=True)
                    cv2.imwrite(f"data/training_visuals/epoch{epoch}_sample{i}.png", combined)

        print("\n[Training Loss Curve]")
        for i, loss in enumerate(losses, 1):
            print(f"Epoch {i}: {loss:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f" Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    print("\n[STEP 1] Generating masks...")
    generate_masks()
    print("\n Mask generation complete. Proceeding to training...\n")
    
    print("[STEP 2] Training segmentation model...")
    train_model()
    print("\n Training complete. Model saved. Proceeding to classification and clustering...\n")
    

    print("\n All done. Ready for temporal inference.")

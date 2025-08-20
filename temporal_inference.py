

import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from PIL import Image
from pathlib import Path
from skimage.measure import find_contours
from skimage import measure
from torchvision import transforms
from models.resnet_segmentor import ResNetSegmentationModel
from models.ct_dataset import improved_wiener_filter

from embedding_utils import ResNetEncoder, compute_topk_similar, extract_feature_vector, compute_similarity, build_support_set_embeddings

# === Added for feature extraction ===
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops

# Initialize deep feature extractor once
deep_encoder = ResNetEncoder().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()


# --- DEBUG OPTIONS ---
DEBUG_SAVE = True
DEBUG_DIR  = "debug_pred"
# --- END DEBUG OPTIONS ---



def apply_clahe(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    return cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)



def visualize_prediction(image, mask, save_path):
    """
    Saves a visualization with image and predicted mask overlay.
    Red contours represent prediction.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.contour(mask, colors='r', linewidths=0.8)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def mask_to_polygon_annotation(mask, image_filename):
    contours = find_contours(mask, 0.5)
    shapes = []
    # ✅ Sort contours by area (largest first)
    filtered_shapes = []
    for contour in contours:
        contour = np.fliplr(contour)  # y, x --> x, y
        if len(contour) < 3:
            continue  # Not a valid polygon

        # Compute area using skimage regionprops
        poly_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.int32(contour)], 1)
        props = measure.regionprops(poly_mask)
        if not props:
            continue
        region = props[0]

        # 🧹 Robust filters to remove noise
        if region.area < 30:
            continue  # ✅ filter very small blobs
        if region.solidity < 0.3:
            continue  # ✅ filter hollow/noisy regions
        if region.eccentricity > 0.995:
            continue  # ✅ filter extremely elongated regions
        # ✅ Relaxed centroid range (5% from borders instead of 10%)
        centroid_y, centroid_x = region.centroid
        h, w = mask.shape
        if not (0.05 * w < centroid_x < 0.95 * w and 0.05 * h < centroid_y < 0.95 * h):
            continue

        # ✅ Add only valid shape
        shape = {
            "label": "predicted",
            "points": contour.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        filtered_shapes.append((region.area, shape))

    # ✅ Keep only the largest shape (if any)
    filtered_shapes.sort(reverse=True, key=lambda x: x[0])
    shapes = [filtered_shapes[0][1]] if filtered_shapes else []
    if not shapes:
        print(f"[ Warning] No significant shapes found in mask: {image_filename}", flush=True)
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_filename),
        "imageData": None,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }
    


# --- Robust threshold helper (Otsu → strict simple fallback) ---

def robust_binarize_from_boosted(boosted: np.ndarray,
                                 min_floor: float = 0.25,   # strict floor
                                 scale_floor: float = 0.60, # strict fraction of max
                                 debug_stem: str = None):
    """
    boosted: float32 [0,1], HxW at model-native resolution (256x256)
    Returns: (binary uint8 mask {0,1}, method string)
    """
    b8 = np.clip(boosted * 255.0, 0, 255).astype(np.uint8)
    thr_255, bin_otsu = cv2.threshold(b8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = float(bin_otsu.sum()) / float(bin_otsu.size + 1e-6)

    used = "otsu"
    bin_mask = bin_otsu

    # If Otsu is basically empty (noise) or too big (bleeding), fallback to a stricter simple cutoff.
    if fg_ratio < 0.0005 or fg_ratio > 0.35:
        m = float(boosted.max())
        t = max(scale_floor * m, min_floor)  # higher threshold → smaller, confident regions
        bin_mask = (boosted > t).astype(np.uint8)
        used = f"fallback_simple(t={t:.4f})"

    if debug_stem:
        try:
            os.makedirs(os.path.dirname(debug_stem), exist_ok=True)
            cv2.imwrite(debug_stem + "_boosted8.png", b8)
            cv2.imwrite(debug_stem + "_bin.png", bin_mask * 255)
        except Exception:
            pass

    return bin_mask, used



def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    bordered = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = bordered.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    flood = flood[1:-1, 1:-1]
    holes = cv2.bitwise_not(flood) & cv2.bitwise_not(m)
    return (((m | holes) > 0).astype(np.uint8))

def largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == best).astype(np.uint8)


def _to_prob_auto(output_tensor):
    out_np = output_tensor.detach().squeeze().cpu().numpy().astype(np.float32)
    mn, mx = float(out_np.min()), float(out_np.max())
    rng = mx - mn
    
    # REPLACE with this — require some dynamic range AND a non-trivial peak:
    already_prob = (0.0 <= mn <= 1.0) and (0.0 <= mx <= 1.0) and (rng > 1e-5) and (mx >= 0.05)
    
    if already_prob:
        return out_np, "activated_head", (mn, mx)
    else:
        prob = 1.0 / (1.0 + np.exp(-out_np))  # sigmoid
        return prob, "logits+sigmoid", (mn, mx)




def run_temporal_inference_on_folder(image_folder, use_clahe: bool = False):
    case_type = os.path.basename(image_folder)  # e.g. "Benign cases"
    label = case_type.replace(" cases", "").lower()
    if not os.path.exists(image_folder):
        print(f" Image folder not found: {image_folder}", flush=True)
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    images_to_infer = image_files
    print(f" [INFO] Inference on {len(images_to_infer)} images in: {image_folder}", flush=True)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSegmentationModel(num_classes=1).to(device)
    ckpt_path = "checkpoints/test_cases_segmentor.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Model not found at: {ckpt_path}", file=sys.stderr, flush=True)
        sys.exit(1)
        
    # === Safer model loading across PyTorch versions ===    
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] {os.path.basename(ckpt_path)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:    print("  missing:", missing[:10], flush=True)
    if unexpected: print("  unexpected:", unexpected[:10], flush=True)
    model.eval()


    # === One-shot embedding support ===
    encoder = ResNetEncoder().to(device).eval()
    support_embeddings = build_support_set_embeddings("data/raw/Test cases")
    

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    os.makedirs("data/masks_png", exist_ok=True)

    for fname in images_to_infer:
        image_path = os.path.join(image_folder, fname)
        image = Image.open(image_path).convert("RGB")
        
        # === Apply Improved Wiener filter ===
        img_np = np.array(image)
        
        # ✅ Optional CLAHE
        if use_clahe:
            img_np = apply_clahe(img_np)

        # Wiener
        img_np = improved_wiener_filter(img_np)

        # ✅ sanitize then clip before uint8 cast
        img_np = np.nan_to_num(img_np, nan=0.0, posinf=255.0, neginf=0.0)
        img_np = np.clip(img_np, 0, 255)
        image = Image.fromarray(img_np.astype(np.uint8))   
        
        # === Step 1: Get embedding and nearest test cases ===
        query_vec = extract_feature_vector(image_path, encoder)
        topk_cases = compute_topk_similar(query_vec, support_embeddings, k=3)
        print(f" Top-k similar test cases for {fname}: {topk_cases}", flush=True)

        # === Step 2: Load mask of nearest test case(s) ===
        mask_accumulator = np.zeros((256, 256), dtype=np.float32)
        valid_count = 0
        for case in topk_cases:
            # case may be a filename with extension; keep just the stem
            base = os.path.splitext(os.path.basename(case))[0]
            p = os.path.join("data", "masks_npy", "Test cases", f"{base}.npy")
            if os.path.exists(p):
                m = np.load(p)
                m = m[0] if m.ndim == 3 else m                # [H,W]
                if m.shape != (256, 256):
                    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask_accumulator += m
                valid_count += 1
            else:
                print(f"[WARN] Guidance mask not found: {p}", flush=True)
     
        if valid_count > 0:
            test_mask = mask_accumulator / valid_count  # average
        else:
            test_mask = np.ones((256, 256), dtype=np.float32)
            print(f"[WARNING] No guidance masks found for {fname}, using default mask.", flush=True)

        # === Step 3: Run model prediction ===
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            # raw_pred = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)
            boosted_raw, head_mode, (mn, mx) = _to_prob_auto(output)

            print(
                f"[PRED] {fname} head={head_mode} "
                f"out[min/max]={mn:.3f}/{mx:.3f} prob[max]={boosted_raw.max():.4f}",
                flush=True
            )

            # === Soft Top-K Mask Boosting ===
            guidance_mask = np.zeros((256, 256), dtype=np.float32)
            for case in topk_cases:
                base = os.path.splitext(os.path.basename(case))[0]
                p = os.path.join("data", "masks_npy", "Test cases", f"{base}.npy")
                if os.path.exists(p):
                    m = np.load(p)
                    m = m[0] if m.ndim == 3 else m
                    if m.shape != (256, 256):
                        m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
                    guidance_mask += (m > 0.5).astype(np.float32)
        
            if len(topk_cases) > 0:
                guidance_mask /= len(topk_cases)  # Normalize guidance

            # --- SAFE BOOSTING ---
            has_guidance = (guidance_mask > 0).any()
            if has_guidance:
                boosted = np.where(guidance_mask > 0, boosted_raw * 1.6, boosted_raw * 0.8)
            else:
                boosted = boosted_raw.copy()  # don't downweight if we have no prior

            boosted = np.clip(boosted, 0, 1)
            # Apply slight Gaussian blur before thresholding
            boosted = cv2.GaussianBlur(boosted, (3, 3), 0)


            # 🔍 Save boosted mask for debugging
            os.makedirs("debug_boosted", exist_ok=True)
            cv2.imwrite(f"debug_boosted/{fname}_boosted.png", (boosted * 255).astype(np.uint8))
            

             
    # ===================== BEGIN PATCH: robust threshold @ 256 =====================
            # Work at model-native 256×256 to avoid upsampled blur lifting the background.
            boosted_256 = boosted

            debug_stem = None
            if DEBUG_SAVE:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                debug_stem = os.path.join(DEBUG_DIR, os.path.splitext(fname)[0])

            binary_256, how = robust_binarize_from_boosted(
                boosted_256,
                min_floor=0.02,   # strict floor (your request: high value)= 0.25
                scale_floor=0.50, # 60% of max= 0.60
                debug_stem=debug_stem
            )

            # Light morphological tidy-up at 256
            binary_256 = cv2.morphologyEx(binary_256, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
            binary_256 = cv2.morphologyEx(binary_256, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            # Keep at 256 throughout to match training + mask generation
            binary = binary_256.copy()

            if DEBUG_SAVE:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{os.path.splitext(fname)[0]}_thresholded.png"),
                            (binary * 255).astype(np.uint8))

            print(f"[INFO] {fname} | boost.max={float(boosted.max()):.4f} | method={how} "
                f"| px256={int(binary_256.sum())}", flush=True)
            
    # ===================== END PATCH: robust threshold @ 256 =====================
    

    # === Begin mask refinement: intersection with high-confidence core ===

            # === Configurable constants (no argparse) ===
            CORE_THR = 0.6       # confidence threshold for core mask
            ERODE_PX = 3         # erosion kernel size in pixels
            ERODE_ITER = 1       # number of erosion iterations
            MIN_AREA = 50        # discard blobs smaller than this
            COMPACTNESS_MIN = 0.1  # minimum compactness to keep blob
            MAX_AREA_FRAC = 0.2    # discard blobs larger than 20% of image
            # =============================================

            # === NEW: Create and erode a high-confidence core ===
            dyn_thr = max(CORE_THR, 0.5 * float(boosted.max()))
            core_mask = (boosted >= dyn_thr).astype(np.uint8)
            if ERODE_PX > 0 and ERODE_ITER > 0:
                kernel = np.ones((ERODE_PX, ERODE_PX), np.uint8)
                core_mask = cv2.erode(core_mask, kernel, iterations=ERODE_ITER)

            # === NEW: Restrict binary mask to components intersecting the core ===
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            keep_labels = set(np.unique(labels[core_mask > 0]))
            refined_mask = np.isin(labels, list(keep_labels)).astype(np.uint8)

            # === NEW: Additional erosion to shrink faint edges further ===
            if ERODE_PX > 0 and ERODE_ITER > 0:
                refined_mask = cv2.erode(refined_mask, kernel, iterations=ERODE_ITER)

            # === NEW: Size and compactness filtering ===
            h, w = refined_mask.shape
            max_area = MAX_AREA_FRAC * h * w
            final_mask = np.zeros_like(refined_mask)
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA or area > max_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                compactness = 4 * np.pi * area / (perimeter**2 + 1e-6)
                if compactness >= COMPACTNESS_MIN:
                    cv2.drawContours(final_mask, [cnt], -1, 1, -1)

            # Fallback if no valid region found
            if final_mask.sum() == 0:
                final_mask = refined_mask if refined_mask.sum() > 0 else binary

            
            binary = final_mask 
            
            # If area is still too huge at 512, clip it again (prevents MAX_AREA_FRAC wipeouts)
            h, w = final_mask.shape
            if final_mask.sum() > 0.35 * h * w:
                # take only the largest compact component
                cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    areas = [(cv2.contourArea(c), c) for c in cnts]
                    areas.sort(reverse=True, key=lambda x: x[0])
                    keep = np.zeros_like(final_mask)
                    cv2.drawContours(keep, [areas[0][1]], -1, 1, -1)
                    final_mask = keep



    # --- Guided component selection ---
    
            # 1) Connected components on the cleaned binary mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels <= 1:
                print(f"[SKIP] No foreground blob found in: {fname}", flush=True)
                continue
            print(f" [DEBUG] Found {num_labels - 1} connected components after cleanup", flush=True)
            
            # 2) Score each component by overlap with the guidance mask, break ties by area
            overlap_scores = []
            guide = (guidance_mask > 0).astype(np.uint8)
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < 30:
                    continue  # drop tiny specks
                comp = (labels == lbl).astype(np.uint8)
                overlap = int((comp * guide).sum())
                overlap_scores.append((overlap, area, lbl))
            if overlap_scores:
                overlap_scores.sort(reverse=True)
                best_overlap, best_area, best_lbl = overlap_scores[0]
                print(f" [DEBUG] Selected component {best_lbl} with overlap {best_overlap} and area {best_area}", flush=True)
            else:
                # fallback: largest area component
                best_lbl = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                sel_area = stats[best_lbl, cv2.CC_STAT_AREA]
                print(f" [DEBUG] Selected component {best_lbl} by area (area={sel_area}) with no guidance overlap", flush=True)
            
            mask = (labels == best_lbl).astype(np.uint8)
            
            # unify to one clean blob
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
            mask = fill_holes(mask)
            mask = largest_component(mask)


            # 3) Light cleanup (morphological open/close already applied earlier)
            print(f" [DEBUG] Non-zero pixels in binary mask: {np.count_nonzero(binary)}", flush=True)
            # Re-check after selection
            if int(mask.sum()) < 30:
                print(f"[SKIP] Low signal in prediction for: {fname}", flush=True)
                continue
            print(f"[PASS] Valid mask extracted for: {fname}", flush=True)
    
    # ---- End guided component selection ---------------
    
    
        # === Save refined overlay using the single-component mask (256×256) ===
    
        case_type   = os.path.basename(image_folder)
        refined_dir = os.path.join("data", "refined_masks", case_type)
        os.makedirs(refined_dir, exist_ok=True)
        refined_path = os.path.join(refined_dir, os.path.splitext(fname)[0] + "_refined.png")

        image_bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlay_img = cv2.resize(image_bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        colored = np.zeros_like(overlay_img)
        colored[:, :, 2] = (mask * 255).astype(np.uint8)  # red
        overlay = cv2.addWeighted(overlay_img, 1.0, colored, 0.5, 0)
        cv2.imwrite(refined_path, overlay)
    
        # === Save final mask and annotations ===

                 

        # Save visual debug composite (input, reference, prediction)
        ref_img_path = os.path.join("data/raw/Test cases", topk_cases[0]) if topk_cases else None
        ref_img = cv2.imread(ref_img_path, cv2.IMREAD_COLOR) if ref_img_path and os.path.exists(ref_img_path) else np.zeros((256,256,3), np.uint8)
        input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        pred_vis = (mask * 255).astype(np.uint8)
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
        input_img = cv2.resize(input_img, (256, 256))
        ref_img   = cv2.resize(ref_img,   (256, 256))
        pred_vis  = cv2.resize(pred_vis,  (256, 256))
        concat = cv2.hconcat([input_img, ref_img, pred_vis])
        debug_dir = "data/visual_debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_debug.png"), concat)
        print(f" [DEBUG] Saved debug image to {os.path.join(debug_dir, fname + '_debug.png')}", flush=True)

        # Convert mask to polygon annotation and save JSON
        json_data = mask_to_polygon_annotation(mask, fname)
        # case_type = os.path.basename(image_path).split()[0] + " cases"  
        case_type = os.path.basename(image_folder)  # e.g., 'Benign cases'      
        annotation_base = os.path.join("data", "annotations", case_type)
        os.makedirs(annotation_base, exist_ok=True)
        json_path = os.path.join(annotation_base, os.path.splitext(fname)[0] + ".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)   
        
        
        # Save .npy
        case_type = os.path.basename(image_folder)  # e.g., 'Benign cases'
        masks_npy_dir = os.path.join("data", "masks_npy", case_type)
        os.makedirs(masks_npy_dir, exist_ok=True)
        npy_path = os.path.join(masks_npy_dir, os.path.splitext(fname)[0] + ".npy")
        np.save(npy_path, mask[None, ...])
        
        
        # Save PNG preview of prediction mask overlay
        case_type = os.path.basename(image_folder)  # e.g., 'Benign cases'
        preview_dir = os.path.join("data", "masks_png", case_type)
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, os.path.splitext(fname)[0] + "_preview.png")
        # visualize_prediction(image, mask, preview_path)
        
        # Convert PIL image to OpenCV BGR for overlay
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_red = np.zeros_like(image_bgr)
        mask_red[:, :, 2] = (mask * 255).astype(np.uint8)  # red channel

        # Overlay with some transparency
        overlay = cv2.addWeighted(image_bgr, 0.7, mask_red, 0.3, 0)

        # Force overwrite preview every run
        cv2.imwrite(preview_path, overlay)
        print(f"[INFO] Updated preview: {preview_path}", flush=True)
        

    # === FEATURE EXTRACTION BLOCK (after final mask) ===
    
        features_dir = Path(f"data/features/inference/{case_type}")
        features_dir.mkdir(parents=True, exist_ok=True)

        # Convert mask to boolean
        mask_bool = mask > 0

        # Make sure the image we crop from matches the mask size (256×256)
        base_img = np.array(image)  # RGB
        if base_img.shape[:2] != mask.shape:
            base_img = cv2.resize(base_img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        if mask_bool.any():
            x, y, w, h = cv2.boundingRect(mask_bool.astype(np.uint8))
            cropped_img  = base_img[y:y+h, x:x+w]   # use base_img, not np.array(image)
            cropped_mask = mask[y:y+h, x:x+w]
        else:
            cropped_img  = base_img                 # aligned with mask
            cropped_mask = mask


        # --- 1. MLDN Features ---
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
        gray_crop = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
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
        deep_feats = extract_feature_vector(image_path, deep_encoder).numpy()

        # Combine all features
        feature_vector = np.concatenate([
            np.array(mldn_feats),
            lbp_hist,
            np.array([entropy_val]),
            np.array(shape_feats),
            deep_feats
        ])

        # Save features
        np.savez(features_dir / f"{Path(image_path).stem}.npz", features=feature_vector)
        print(f"[OK] Saved features: {features_dir / (Path(image_path).stem + '.npz')}")
    
    # === End feature extraction ===
        

if __name__ == "__main__":
    import traceback
    try:
        parser = argparse.ArgumentParser(description="Run temporal inference and save polygon annotations.")
        parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder (e.g. data/raw/Benign cases)")
        parser.add_argument("--use_clahe", action="store_true", help="Apply CLAHE during preprocessing.")
        args = parser.parse_args()
        run_temporal_inference_on_folder(args.image_dir, use_clahe=args.use_clahe)
   
        print(" Inference running on folder:", args.image_dir, flush=True)
        
    except Exception as e:
        print(" Exception in temporal_inference.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)

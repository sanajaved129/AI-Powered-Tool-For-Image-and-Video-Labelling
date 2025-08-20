

from PIL import Image, ImageDraw
import json
import numpy as np
import os


def convert_all(json_dir, png_output_dir, npy_output_dir, raw_image_folder):
    
    label_folder = os.path.basename(json_dir).lower()
    
    case_folder = {
        "benign": "Benign cases", "malignant": "Malignant cases",
        "normal": "Normal cases", "test": "Test cases"
    }[label_folder]

    out_png_dir = os.path.join(png_output_dir, case_folder)
    out_npy_dir = os.path.join(npy_output_dir, case_folder)
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_npy_dir, exist_ok=True)


    label_mapping = {
        "benign": "Benign cases",
        "malignant": "Malignant cases",
        "normal": "Normal cases",
        "test": "Test cases"
    }

    if label_folder not in label_mapping:
        print(f"❌ Unknown label folder: {label_folder}")
        return

    image_subfolder = os.path.join(raw_image_folder, label_mapping[label_folder])

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, filename)
        image_filename = None
        image_size = None

        for ext in [".jpg", ".jpeg", ".png"]:
            test_path = os.path.join(image_subfolder, filename.replace(".json", ext))
            if os.path.exists(test_path):
                image_filename = os.path.splitext(os.path.basename(test_path))[0]
                with Image.open(test_path) as img:
                    image_size = img.size  # ✅ (width, height)
                # image_size = Image.open(test_path).size[::-1]  # (H, W)
                break

        if not image_filename:
            print(f"⚠️ Could not find matching image for {filename}")
            continue

        # Load annotation
        with open(json_path, "r") as f:
            data = json.load(f)

        mask_img = Image.new("L", image_size, 0)
        # mask_img = Image.new("L", image_size[::-1], 0)  # size = (width, height)
        draw = ImageDraw.Draw(mask_img)

        pts = data.get("points", [])
        coord_space = int(data.get("coord_space", 0) or 0)

        # Scale to original image size if points are in 256-space
        if coord_space == 256:
            w, h = image_size  # (width, height)
            points = [(int(round(x * w / 256.0)), int(round(y * h / 256.0))) for x, y in pts]
        else:
            points = [tuple(pt) for pt in pts]

        if data.get("mode") == "Polygon":
            if len(points) >= 3:
                draw.polygon(points, fill=1)
        elif data.get("mode") == "Freehand":
            if len(points) >= 2:
                draw.line(points, fill=1, width=3)


        mask = np.array(mask_img)  # Now assign drawn image to the mask


        # Save PNG
        png_path = os.path.join(png_output_dir, f"{image_filename}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(png_path)

        # Resize + Save NPY
        resized_mask = mask_img.resize((256, 256), resample=Image.NEAREST)
        normalized_mask = (np.array(resized_mask) > 0).astype(np.float32)  # 0 or 1
        
        # Save as .npy
        npy_path = os.path.join(npy_output_dir, f"{image_filename}.npy")
        np.save(npy_path, normalized_mask[np.newaxis, ...])
        


        print(f"✅ Saved: {png_path} and .npy")




# Run it
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("JSON → mask converter")
    ap.add_argument("--json_dir", required=True, help="e.g. data/annotations/benign")
    ap.add_argument("--png_out",  default="data/masks_png")
    ap.add_argument("--npy_out",  default="data/masks_npy") 
    ap.add_argument("--raw",      default="data/raw")
    args = ap.parse_args()
    convert_all(args.json_dir, args.png_out, args.npy_out, args.raw)

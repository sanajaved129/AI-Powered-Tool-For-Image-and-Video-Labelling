

import os, json
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import cv2

from models.resnet_segmentor import ResNetSegmentationModel  


# ---------------- Path normalizer ----------------

RAW_ROOT = os.path.join("data", "raw")

def _stem(path: str) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    # strip preview suffix if user clicked a preview
    return s[:-9] if s.endswith("_preview") else s

def _case_from_path(path: str):
    p = path.replace("\\", "/")
    # try to read case folder from known roots
    for key in ("/raw/", "/masks_png/", "/masks_npy/", "/annotations/"):
        if key in p:
            # ".../<root>/<case>/<file>"
            try:
                return p.split(key, 1)[1].split("/", 1)[0]
            except Exception:
                pass
    return None

def _find_raw_image(stem: str, case: str = None):
    """
    Find the raw CT image file for a given stem (and optional case folder).
    Searches data/raw/<case>/<stem>.{jpg,jpeg,png} first, then all cases.
    """
    exts = (".jpg", ".jpeg", ".png")
    # case-specific search
    if case:
        for ext in exts:
            cand = os.path.join(RAW_ROOT, case, stem + ext)
            if os.path.exists(cand):
                return cand
    # global search across cases
    if os.path.isdir(RAW_ROOT):
        for d in os.listdir(RAW_ROOT):
            for ext in exts:
                cand = os.path.join(RAW_ROOT, d, stem + ext)
                if os.path.exists(cand):
                    return cand
    # final fallback: same dir as given path (if user already browsed raw)
    for ext in exts:
        cand = stem + ext
        if os.path.exists(cand):
            return cand
    return None

def normalize_to_raw(image_path: str) -> str:
    """
    Map any selected path (preview png / npy / json / raw) to the RAW CT image path.
    """
    s = _stem(image_path)
    case = _case_from_path(image_path)
    raw = _find_raw_image(s, case)
    return raw if raw else image_path  # if not found, keep the original (won't crash)

# ---------- Thresholding ----------

def robust_binarize_from_prob(prob: np.ndarray,
                              min_floor: float = 0.25,
                              scale_floor: float = 0.60):
    """
    prob: float32 [H,W] in [0,1]
    Returns: bin (uint8 {0,1}), method string
    """
    b8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    _, bin_otsu = cv2.threshold(b8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = float(bin_otsu.sum()) / float(bin_otsu.size + 1e-6)

    used = "otsu"
    bin_mask = bin_otsu
    if fg_ratio < 0.0005 or fg_ratio > 0.35:
        m = float(prob.max())
        t = max(scale_floor * m, min_floor)
        bin_mask = (prob > t).astype(np.uint8)
        used = f"fallback_simple(t={t:.3f})"

    k = np.ones((3,3), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  k)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k)
    return bin_mask, used

# ---------- JSON → mask ----------

def rasterize_json_polygon(json_path: str, size=(256, 256)) -> np.ndarray:
    """
    Reads either:
      - LabelMe-style: {"shapes":[{"points":[[x,y],...]}, ...]}
      - GUI-style:     {"points":[[x,y],...], "coord_space":256}
    Returns a (H,W) uint8 mask in {0,1}.
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.uint8)

    with open(json_path, "r") as f:
        data = json.load(f)

    # 1) Try LabelMe first
    shapes = data.get("shapes")
    if isinstance(shapes, list) and shapes:
        pts = shapes[0].get("points", [])
    else:
        # 2) Fallback to GUI top-level format
        pts = data.get("points", [])
        if int(data.get("coord_space", 0) or 0) == 256:
            # map 256->target size
            pts = [[int(round(x * W / 256.0)), int(round(y * H / 256.0))] for x, y in pts]

    if len(pts) < 3:
        return mask

    cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [cnt], 1)
    return mask


# ---------- Overlay (uses RAW image) ----------

def make_overlay(image_path: str, mask_uint8: np.ndarray, out_size=512) -> Image.Image:
    """
    Builds a semi-transparent red overlay for display.
    Always uses the RAW CT image that corresponds to image_path.
    """
    raw = normalize_to_raw(image_path)
    img = Image.open(raw).convert("RGB")
    base = ImageOps.autocontrast(img.resize((out_size, out_size)))

    opacity = 0.35  # nicer for reading CT behind mask
    alpha = (mask_uint8.astype(np.uint8) * int(255 * opacity))
    alpha = Image.fromarray(alpha).resize((out_size, out_size), resample=Image.NEAREST)

    overlay = Image.new("RGBA", (out_size, out_size), (255, 0, 0, 0))
    overlay.putalpha(alpha)
    return Image.alpha_composite(base.convert("RGBA"), overlay)
  
  

# ---------- Load predicted mask produced by temporal_inference ----------

def load_pred_mask_from_outputs(image_path: str):
    """
    Returns a binary uint8 mask (256x256) for the given RAW image,
    using artifacts saved by temporal_inference.py.

    Search order:
      1) data/masks_npy/<Case>/<stem>.npy   (preferred)
      2) data/annotations/<Case>/<stem>.json  (rasterize polygon)
    Returns None if not found.
    """
    stem = _stem(image_path)
    case = _case_from_path(image_path)

    # 1) NPY under case folder (what temporal_inference.py writes)
    if case:
        p = os.path.join("data", "masks_npy", case, f"{stem}.npy")
        if os.path.exists(p):
            m = np.load(p)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            m = (m > 0.5).astype(np.uint8)
            if m.shape != (256, 256):
                m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
            return m

    # 2) JSON polygon under case (temporal_inference.py also writes this)
    if case:
        j = os.path.join("data", "annotations", case, f"{stem}.json")
        if os.path.exists(j):
            return rasterize_json_polygon(j, size=(256, 256))

    # (Optional) widen search a bit if needed:
    # - root-level masks_npy/<stem>.npy
    p_root = os.path.join("data", "masks_npy", f"{stem}.npy")
    if os.path.exists(p_root):
        m = np.load(p_root)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        m = (m > 0.5).astype(np.uint8)
        if m.shape != (256, 256):
            m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
        return m

    # - annotations/<label>/<stem>.json
    label = (case or "").replace(" cases", "").lower()
    if label:
        j2 = os.path.join("data", "annotations", label, f"{stem}.json")
        if os.path.exists(j2):
            return rasterize_json_polygon(j2, size=(256, 256))

    return None  



# ---------- Predictor (predicts on RAW image) ----------

class Predictor:
    def __init__(self, ckpt_path, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) build the model FIRST
        self.model = ResNetSegmentationModel(num_classes=1).to(self.device).eval()

        # 2) use the path AS-IS (full path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # 3) load checkpoint
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[CKPT] loaded {ckpt_path}  missing={len(missing)} unexpected={len(unexpected)}")

        # 4) inference transforms
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def predict(self, image_path: str):
        """
        Returns:
          prob: float32 (256,256) in [0,1]
          bin_mask: uint8 (256,256) in {0,1}
          how: thresholding method used
        """
        raw = normalize_to_raw(image_path)  # <— ensure we always use the raw CT
        img = Image.open(raw).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            prob = torch.sigmoid(out).squeeze().cpu().numpy().astype(np.float32)
        bin_mask, how = robust_binarize_from_prob(prob)
        return prob, bin_mask, how
      
    
    
    def predict_prob_256(self, image_path: str) -> np.ndarray:
        """
        Run the model on a raw image and return a 256×256 probability map in [0,1].
        Ensures a 3-channel RGB tensor for models whose first conv expects 3 channels.
        """
        # Always infer from the RAW CT and use RGB
        raw = normalize_to_raw(image_path)
        img = Image.open(raw).convert("RGB")

        x = self.transform(img).unsqueeze(0).to(self.device)  # [1, C, 256, 256]
        print(f"[DBG] input x mean={x.mean().item():.4f} std={x.std().item():.4f}")

        # Safety: if for any reason C==1, replicate to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        with torch.no_grad():
            out = self.model(x)  # logits or probs
            out_np = out.squeeze().detach().cpu().numpy().astype(np.float32)

        # Convert logits → probs if needed
        mn, mx = float(out_np.min()), float(out_np.max())
        already_prob = (0.0 <= mn <= 1.0) and (0.0 <= mx <= 1.0) and (mx - mn > 1e-6) and (mx >= 0.05)
        prob = out_np if already_prob else 1.0 / (1.0 + np.exp(-out_np))

        # Ensure shape/bounds
        if prob.shape != (256, 256):
            prob = cv2.resize(prob, (256, 256), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0)

    

# ---------- Evaluator (GT source selectable, prediction input is RAW) ----------

class Evaluator:
    """
    source ∈ {'NPY','PNG','JSON'} — where to read GT from.
    Prediction is always run on the RAW CT image (via normalize_to_raw).
    """
    def __init__(self, source: str):
        src = source.upper()
        if src not in {"NPY", "PNG", "JSON"}:
            raise ValueError("source must be one of 'NPY','PNG','JSON'")
        self.source = src

    @staticmethod
    def dice(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        pred = pred.astype(np.uint8).ravel()
        tgt  = target.astype(np.uint8).ravel()
        inter = (pred & tgt).sum()
        return (2*inter + eps) / (pred.sum() + tgt.sum() + eps)

    @staticmethod
    def iou(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        pred = pred.astype(np.uint8).ravel()
        tgt  = target.astype(np.uint8).ravel()
        inter = (pred & tgt).sum()
        union = pred.sum() + tgt.sum() - inter
        return (inter + eps) / (union + eps)
      
    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        pred = pred.astype(np.uint8)
        tgt  = target.astype(np.uint8)
        tp = (pred & tgt).sum()
        fp = (pred & (1 - tgt)).sum()
        return (tp + eps) / (tp + fp + eps)

    @staticmethod
    def accuracy(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        pred = pred.astype(np.uint8)
        tgt  = target.astype(np.uint8)
        tp = (pred & tgt).sum()
        tn = ((1 - pred) & (1 - tgt)).sum()
        fp = (pred & (1 - tgt)).sum()
        fn = ((1 - pred) & tgt).sum()
        return (tp + tn + eps) / (tp + tn + fp + fn + eps)

    
    

    # ----- GT finders (prefer matching case folder if known) -----

    def _find_gt_npy(self, stem: str, case: str = None):
        # case-specific first
        root = os.path.join("data", "masks_npy")
        if case and os.path.isdir(root):
            p = os.path.join(root, case, f"{stem}.npy")
            if os.path.exists(p): return p
        # root-level
        p = os.path.join(root, f"{stem}.npy")
        if os.path.exists(p): return p
        # nested fallback
        if os.path.isdir(root):
            for d in os.listdir(root):
                q = os.path.join(root, d, f"{stem}.npy")
                if os.path.exists(q): return q
        return None

    def _find_gt_png(self, stem: str, case: str = None):
        root = os.path.join("data", "masks_png")
        # case-specific binary (not preview)
        if case and os.path.isdir(root):
            q = os.path.join(root, case, f"{stem}.png")
            if os.path.exists(q) and not q.endswith("_preview.png"):
                return q
        # root-level
        q = os.path.join(root, f"{stem}.png")
        if os.path.exists(q) and not q.endswith("_preview.png"):
            return q
        # nested fallback
        if os.path.isdir(root):
            for d in os.listdir(root):
                q = os.path.join(root, d, f"{stem}.png")
                if os.path.exists(q) and not q.endswith("_preview.png"):
                    return q
        return None

    def _find_gt_json(self, stem: str, case: str = None):
        root = os.path.join("data", "annotations")
        # 1) case folder (e.g., "Benign cases")
        if case and os.path.isdir(root):
            p = os.path.join(root, case, f"{stem}.json")
            if os.path.exists(p): return p
        # 2) label folder (e.g., "benign")
        if case:
            label = case.replace(" cases", "").lower()
            q = os.path.join(root, label, f"{stem}.json")
            if os.path.exists(q): return q
        # 3) fallback: search subfolders
        if os.path.isdir(root):
            for d in os.listdir(root):
                r = os.path.join(root, d, f"{stem}.json")
                if os.path.exists(r): return r
        return None


    def load_ground_truth(self, image_path: str) -> np.ndarray:
        """
        Returns binary GT mask (256×256, uint8 {0,1}) or None if missing.
        Uses the stem/case derived from the RAW image path.
        """
        raw = normalize_to_raw(image_path)
        stem = _stem(raw)
        case = _case_from_path(raw)

        if self.source == "NPY":
            npy = self._find_gt_npy(stem, case)
            if not npy:
                return None
            m = np.load(npy)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            m = (m > 0.5).astype(np.uint8)
            if m.shape != (256, 256):
                m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
            return m

        if self.source == "PNG":
            png = self._find_gt_png(stem, case)
            if not png:
                return None
            m = np.array(Image.open(png).convert("L"))
            m = (m > 127).astype(np.uint8)
            if m.shape != (256, 256):
                m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
            return m

        if self.source == "JSON":
            j = self._find_gt_json(stem, case)
            if not j:
                return None
            m = rasterize_json_polygon(j, size=(256, 256))
            return m

        return None


    def evaluate_one(self, pred_mask_256: np.ndarray, image_path: str):
        gt = self.load_ground_truth(image_path)
        if gt is None:
            return None, None, None, None
        d = self.dice(pred_mask_256, gt)
        j = self.iou(pred_mask_256, gt)
        p = self.precision(pred_mask_256, gt)
        a = self.accuracy(pred_mask_256, gt)
        return d, j, p, a

      
    
   


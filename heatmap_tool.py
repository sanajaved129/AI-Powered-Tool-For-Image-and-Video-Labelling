
"""
Optional probability heatmap utility for research/debugging.
- Uses existing Predictor from inference_eval.py (256×256).
- Returns a 512×512 RGBA overlay for display or saving.
- Can be used from GUI or CLI.


"""


import os, argparse
import numpy as np
from PIL import Image, ImageOps
import cv2
import matplotlib.cm as cm
from inference_eval import Predictor




# ---------------- internal helpers ----------------

def _cv2_colormap_id(name: str):
    name = (name or "").strip().lower()
    table = {
        "jet": cv2.COLORMAP_JET,
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "hot": cv2.COLORMAP_HOT,
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
    }
    return table.get(name, cv2.COLORMAP_JET)

def _prob_to_colormap(prob_256: np.ndarray, colormap: str, out_size: int) -> np.ndarray:
    """prob in [0,1], returns BGR heatmap at out_size×out_size."""
    p8  = (np.clip(prob_256, 0, 1) * 255).astype(np.uint8)
    pR  = cv2.resize(p8, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    cmap_id = _cv2_colormap_id(colormap)
    return cv2.applyColorMap(pR, cmap_id)


# ---------------- public API ----------------

def build_heatmap_overlay(image_path: str,
                          predictor: Predictor = None,
                          ckpt_path: str = None,
                          colormap: str = "jet",
                          alpha: float = 0.40,
                          out_size: int = 512,
                          thr_abs: float = 0.60,
                          gamma: float = 1.2,
                          soften: int = 3,
                          top_percent: float = 0.5,     # used only for flat maps
                          flat_color=(255, 0, 0)):      # red for flat-map fallback
    """
    Color only where prob >= thr_abs when the map has decent dynamic range.
    If the map is nearly flat (like 0.5 everywhere), color only the TOP <top_percent>%
    of pixels with a constant alpha so hotspots are actually visible.
    """

    if predictor is None:
        if not ckpt_path:
            ckpt_path = os.path.join("checkpoints", "test_cases_segmentor.pth")
        predictor = Predictor(ckpt_path)

    prob = predictor.predict_prob_256(image_path).astype(np.float32)  # [256,256] in [0,1]

    if soften > 0:
        k = 2 * soften + 1
        prob = cv2.GaussianBlur(prob, (k, k), 0)

    pmin, pmax = float(prob.min()), float(prob.max())
    dyn = pmax - pmin

    use_flat_fallback = (dyn < 0.02)  # your case: ~0.003–0.008

    if not use_flat_fallback:
        # ---------- NORMAL CASE (decent dynamic range) ----------
        thr = float(thr_abs)
        ramp = np.clip((prob - thr) / (1.0 - thr + 1e-6), 0.0, 1.0)
        if gamma != 1.0:
            ramp = ramp ** float(gamma)
        alpha_map = (ramp * alpha).astype(np.float32)

        cmap = cm.get_cmap(colormap)
        heat_rgb = (cmap(prob)[..., :3] * 255).astype(np.uint8)

        heat_a = (alpha_map * 255).astype(np.uint8)
        heat_rgba = np.dstack([heat_rgb, heat_a])

        mode_used = f"abs thr={thr:.3f}"

    else:
        # ---------- FLAT-MAP FALLBACK (make hotspots visible) ----------
        # keep ONLY the top X% pixels; constant alpha for those pixels
        q = 1.0 - (top_percent / 100.0)               # e.g., 0.995 for top 0.5%
        thr = float(np.quantile(prob, q))
        mask = (prob >= thr).astype(np.uint8)

        # optional dilation so a few pixels become visible blobs
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        heat_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        heat_rgba[..., 0] = flat_color[0]
        heat_rgba[..., 1] = flat_color[1]
        heat_rgba[..., 2] = flat_color[2]
        heat_rgba[..., 3] = (alpha * 255.0 * mask).astype(np.uint8)

        mode_used = f"pct top={top_percent:.3f} thr={thr:.4f}"

    # Upscale to display size
    heat_rgba = cv2.resize(heat_rgba, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # Base RGBA
    base = Image.open(image_path).convert("L").resize((out_size, out_size), Image.BILINEAR)
    base_rgba = Image.merge("RGBA", (base, base, base, Image.new("L", base.size, 255)))

    # Composite
    heat_img = Image.fromarray(heat_rgba, mode="RGBA")
    out = Image.alpha_composite(base_rgba, heat_img)

    colored_px = int((heat_rgba[..., 3] > 0).sum())
    print(f"[Heatmap] {os.path.basename(image_path)} | prob[min/mean/max]={pmin:.4f}/{float(prob.mean()):.4f}/{pmax:.4f} "
          f"| {mode_used} | colored_px={colored_px}")
    return out





def save_heatmap(image_path: str, out_path: str, **kwargs):
    """Convenience save."""
    overlay = build_heatmap_overlay(image_path, **kwargs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    overlay.save(out_path)
    print(f"[OK] Saved heatmap → {out_path}")

# ---------------- CLI ----------------

def _run_single(args):
    pred = Predictor(args.ckpt)
    save_heatmap(args.image, args.out, predictor=pred,
                 colormap=args.colormap, alpha=args.alpha, out_size=args.size)

def _run_dir(args):
    pred = Predictor(args.ckpt)
    images = [f for f in os.listdir(args.dir)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    os.makedirs(args.outdir, exist_ok=True)
    for f in images:
        ip = os.path.join(args.dir, f)
        op = os.path.join(args.outdir, os.path.splitext(f)[0] + "_heatmap.png")
        try:
            save_heatmap(ip, op, predictor=pred,
                         colormap=args.colormap, alpha=args.alpha, out_size=args.size)
        except Exception as e:
            print(f"[WARN] {f}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Heatmap tool (optional)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("image", help="single image")
    s1.add_argument("--image", required=True)
    s1.add_argument("--ckpt",  default="test_cases_segmentor.pth")
    s1.add_argument("--out",   default="heatmap.png")
    s1.add_argument("--colormap", default="jet")
    s1.add_argument("--alpha", type=float, default=0.40)
    s1.add_argument("--size",  type=int, default=512)
    s1.set_defaults(func=_run_single)

    s2 = sub.add_parser("dir", help="batch folder")
    s2.add_argument("--dir",   required=True)
    s2.add_argument("--ckpt",  default="test_cases_segmentor.pth")
    s2.add_argument("--outdir", default="heatmaps")
    s2.add_argument("--colormap", default="jet")
    s2.add_argument("--alpha", type=float, default=0.40)
    s2.add_argument("--size",  type=int, default=512)
    s2.set_defaults(func=_run_dir)

    args = ap.parse_args()
    args.func(args)

# src/scripts/depth_refine_masks.py
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.models.depth_estimator import DepthEstimator

def edge_thin_mask(mask01, strength_px=2):
    """لبه‌محور: ماسک اشیا را نازک می‌کند تا سوراخ‌های باریک از بین نروند."""
    m = (mask01 > 0).astype(np.uint8)
    edges = cv2.Canny(m*255, 50, 150)
    if strength_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (strength_px*2+1, strength_px*2+1))
        edges = cv2.dilate(edges, k, 1)
    # هرجا edge هست، از ماسک کم کنیم تا هاله‌ها کم بشن
    m = cv2.subtract(m*255, edges)
    return (m > 127).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cpu", help="cpu یا cuda")
    ap.add_argument("--save-vis", default="", help="پریویوی بصری اصلاح ماسک‌ها")
    ap.add_argument("--depth-scale", type=float, default=0.06,
                    help="آستانه‌ی نزدیکی نسبت به عمق دیوار؛ بزرگ‌تر = اشیای بیشتری جلو حساب می‌شود")
    ap.add_argument("--edge-thin", type=int, default=2, help="شدت نازک‌سازی لبه‌ها (0=خاموش)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    rooms = Path(cfg["paths"]["rooms"])
    wall_masks = Path(cfg["paths"]["wall_masks"])
    obj_masks  = Path(cfg["paths"]["obj_masks"])

    out_vis = Path(args.save_vis) if args.save_vis else None
    if out_vis:
        out_vis.mkdir(parents=True, exist_ok=True)

    depth_net = DepthEstimator(device=args.device)

    room_files = sorted([p for p in rooms.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])

    for rp in tqdm(room_files, desc="Depth-aware refine"):
        wm = wall_masks / (rp.stem + ".png")
        om = obj_masks  / (rp.stem + ".png")

        # بخوان
        bgr = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        wall = cv2.imread(str(wm), cv2.IMREAD_GRAYSCALE)
        if wall is None:
            H,W = bgr.shape[:2]
            wall = np.zeros((H,W), np.uint8)

        obj = cv2.imread(str(om), cv2.IMREAD_GRAYSCALE)
        if obj is None:
            obj = np.zeros_like(wall, np.uint8)

        # عمق
        depth_inv = depth_net.predict(rgb)  # [0..1] بزرگ = نزدیک
        H,W = wall.shape

        # عمق دیوار = میانه‌ی عمق داخل ماسک دیوارِ فعلی
        wall_pos = depth_inv[wall > 127]
        if wall_pos.size < 50:
            # اگر ماسک دیوار خیلی کوچک بود، از نیمه‌ی بالایی تصویر حدس بزن
            wall_pos = depth_inv[0:int(H*0.55), :]
        wall_med = float(np.median(wall_pos))

        # هر پیکسلی که به‌طور معنی‌دار نزدیک‌تر از دیوار باشد → شیء جلو
        near = (depth_inv >= wall_med + args.depth_scale).astype(np.uint8)

        # اشیای جدید = near داخل محدوده‌ی دیوار یا نزدیک مرز دیوار
        # تا مطمئن شیم دیوار رو سوراخ می‌کنیم جایی که باید
        wall_dil = cv2.dilate((wall>127).astype(np.uint8), np.ones((7,7),np.uint8), 1)
        obj_new = (near & wall_dil).astype(np.uint8)

        # فیوژن با ماسک اشیای قبلی
        obj_fused = np.clip(obj//255 + obj_new, 0, 1).astype(np.uint8)

        # نازک‌سازی لبه برای حفظ خطوط باریک
        if args.edge_thin > 0:
            obj_fused = edge_thin_mask(obj_fused, strength_px=args.edge_thin)

        # تمیزکاری
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        obj_fused = cv2.morphologyEx(obj_fused, cv2.MORPH_OPEN, k, iterations=1)

        # دیوار جدید = دیوار قدیم منهای اشیا
        wall_new = ((wall>127).astype(np.uint8) & (1 - obj_fused)).astype(np.uint8)

        # ذخیره
        cv2.imwrite(str(om), (obj_fused*255).astype(np.uint8))
        cv2.imwrite(str(wm), (wall_new*255).astype(np.uint8))

        if out_vis:
            overlay = bgr.copy()
            # قرمز = دیوار، سبز = شیء
            overlay[wall_new==1] = (0,0,255)
            overlay[obj_fused==1] = (0,255,0)
            vis = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
            cv2.imwrite(str(out_vis / f"{rp.stem}__depth_refine.jpg"), vis)

if __name__ == "__main__":
    main()

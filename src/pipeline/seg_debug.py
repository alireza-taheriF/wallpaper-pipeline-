# src/pipeline/seg_debug.py
import numpy as np
from pathlib import Path

def save_gray(path, arr):
    from PIL import Image
    a = arr
    a = (255*(a - a.min())/max(1e-6, a.max()-a.min())).astype(np.uint8)
    Image.fromarray(a).save(path)

def colorize_labels(lbl):
    import numpy as np
    import cv2
    # پالِت ساده 256 رنگ
    rng = np.random.default_rng(42)
    palette = (rng.integers(0,256,(256,3))).astype(np.uint8)
    return palette[lbl % 256]

def dump_seg_debug(base, img_rgb, logits_or_labels, label_to_index, wall_class_names, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # نرمال‌سازی ورودی‌های متنوع: یا لاجیت [C,H,W]/[1,C,H,W] یا لیبل [H,W]
    arr = logits_or_labels
    if arr.ndim == 4 and arr.shape[0] == 1: arr = arr[0]          # [C,H,W]
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:            # [C,H,W]
        C,H,W = arr.shape
        logits = arr
        # softmax
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = e / np.clip(e.sum(axis=0, keepdims=True), 1e-8, None)
        labels = probs.argmax(axis=0).astype(np.int32)
    elif arr.ndim == 2:
        probs = None
        labels = arr.astype(np.int32)
        C,H,W = 0, labels.shape[0], labels.shape[1]
    else:
        raise RuntimeError(f"Unsupported seg output shape: {arr.shape}")

    # دیوار: بیشینهٔ احتمال بین wall کلاس‌ها
    wall_ids = [label_to_index[c] for c in wall_class_names if c in label_to_index]
    if wall_ids and probs is not None:
        wall_prob = probs[wall_ids].max(axis=0)
    else:
        wall_prob = (labels == label_to_index.get("wall", -999)).astype(np.float32)

    # ذخیرهٔ دیباگ
    from PIL import Image
    Image.fromarray(colorize_labels(labels)).save(out/f"{base}.seg_labels.png")
    save_gray(out/f"{base}.wall_prob.png", wall_prob)
    # overlay
    import cv2
    img = img_rgb[:, :, ::-1].copy()  # RGB->BGR برای cv2
    overlay = (0.5*img + 0.5*cv2.applyColorMap((wall_prob*255).astype(np.uint8), cv2.COLORMAP_JET)).astype(np.uint8)
    cv2.imwrite(str(out/f"{base}.overlay.png"), overlay)

    # پرینت آمار
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[SEG] base={base} labels.unique={dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"[SEG] wall_prob stats: min={float(wall_prob.min()):.3f} max={float(wall_prob.max()):.3f} mean={float(wall_prob.mean()):.3f}")
    return labels, wall_prob



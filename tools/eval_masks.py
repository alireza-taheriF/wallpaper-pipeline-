# tools/eval_masks.py
import cv2, os
from pathlib import Path

NAMES = {
 "download_14b8c046-93f8-4822-801e-2334d9dca696",
 "download_1569ffce-8b33-4a0f-a73e-5267e57fc61d",
 "download_1e0c70f6-e691-4e21-95f6-336d60f9619d",
}
masks = Path("src/data/out/masks")
reports = Path("src/data/out/reports")
reports.mkdir(parents=True, exist_ok=True)

rows = []
for base in sorted(NAMES):
    w = cv2.imread(str(masks/f"{base}.wall.png"), 0)
    o = cv2.imread(str(masks/f"{base}.objects.png"), 0)
    def stat(m):
        if m is None: return ("NA", 0, 0)
        nz = int((m>0).sum()); tot = m.shape[0]*m.shape[1]
        return (f"{nz/tot:.3f}", nz, tot)
    wr, wnz, wtot = stat(w); or_, onz, otot = stat(o)
    rows.append([base, wr, wnz, wtot, or_, onz, otot])

with open(reports/"coverage.csv","w") as f:
    f.write("file,wall_ratio,wall_nonzero,wall_total,obj_ratio,obj_nonzero,obj_total\n")
    for r in rows: f.write(",".join(map(str,r))+"\n")

print("Wrote", reports/"coverage.csv")

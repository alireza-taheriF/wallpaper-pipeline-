# tools/log_signals.py
from pathlib import Path
log = Path("src/data/out/reports/smoke_test.log").read_text()
def count(s): return log.count(s)
print({
 "too_small": count("Wall mask too small"),
 "no_contours": count("No contours found in wall mask"),
 "fallback_rect": count("minimal area rectangle"),
 "no_wall_pixels": count("No wall pixels found for lightness extraction"),
})

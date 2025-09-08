import hashlib, glob, os
for p in sorted(glob.glob("src/data/out/masks/*.objects.png")):
    b = open(p,"rb").read()
    print(os.path.basename(p), hashlib.md5(b).hexdigest(), len(b))


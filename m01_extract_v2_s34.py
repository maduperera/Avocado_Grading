"""
Extraction v2 (sides 3 & 4)
  * flat-field calibration, pixel-wise:  R = I / max(Ref_same_channel_same_angle, 10)
  * ONE spot centre per fruit-side, shared by r,g,b: brightest point of the angle-averaged, blurred GREEN frame
  * truncated-spot flag: centre closer than 40 px to the image border (usable area = top 85 % rows)
  * radial profile (mean R in rings around the centre) per channel & angle:
        rings (px): 0-10, 10-20, 20-30, 30-40, 40-60, 60-80      (+ 'box75' = old-style 75x75 box at the centre)
Output: analysis_output_m01_v2/spots_meta.csv, ring_long.csv
"""
import os, glob, cv2, numpy as np, pandas as pd
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(BASE, "Malwatte_01")
OUT = os.path.join(BASE, "analysis_output_m01_v2"); os.makedirs(OUT, exist_ok=True)
H, W = 240, 320; HU = int(H * .85)
EDGES = [0, 10, 20, 30, 40, 60, 80]; NR = len(EDGES) - 1
REF = {}
for fp in glob.glob(os.path.join(BASE, "Ref", "*.png")):
    p = os.path.basename(fp).split("_"); REF[(p[0], int(p[1]))] = fp
_refc = {}
yy, xx = np.mgrid[0:HU, 0:W]


def load(fp): return cv2.imread(fp, 0)[:HU].astype(np.float32)


def ref(ch, a):
    k = (ch, a)
    if k not in _refc: _refc[k] = np.maximum(load(REF[k]), 10.0)
    return _refc[k]


def job(args):
    cls, side, fid, folder = args
    files = {}
    for fp in glob.glob(os.path.join(folder, "*.png")):
        p = os.path.basename(fp).split("_")
        if len(p) >= 4 and p[0] in "rgb": files[(p[0], int(p[1]))] = fp
    angs = sorted(a for (c, a) in files if c == "g")
    if not angs: return None, []
    G = {a: load(files[("g", a)]) for a in angs}
    mean_g = np.mean(list(G.values()), axis=0)
    bl = cv2.GaussianBlur(mean_g, (0, 0), 7); _, _, _, (cx, cy) = cv2.minMaxLoc(bl)
    trunc = int(cx < 40 or cx > W - 40 or cy < 40 or cy > HU - 40)
    d = np.hypot(xx - cx, yy - cy); ridx = np.digitize(d, EDGES) - 1; ridx[(ridx < 0) | (ridx >= NR)] = NR  # NR = outside
    cnt = np.bincount(ridx.ravel(), minlength=NR + 1)[:NR]
    l, t = max(0, cx - 37), max(0, cy - 37); r_, b_ = min(W, l + 75), min(HU, t + 75)
    rows = []
    for ch in "rgb":
        for a in angs:
            if (ch, a) not in files or (ch, a) not in REF: continue
            im = G[a] if ch == "g" else load(files[(ch, a)])
            R = im / ref(ch, a)
            s = np.bincount(ridx.ravel(), weights=R.ravel(), minlength=NR + 1)[:NR] / np.maximum(cnt, 1)
            rows.append((cls, side, fid, ch, a, *s, float(R[t:b_, l:r_].mean()), float(im[t:b_, l:r_].mean())))
    meta = dict(cls=cls, side=side, fid=fid, cx=int(cx), cy=int(cy), trunc=trunc, nang=len(angs), peak=float(bl.max()))
    return meta, rows


if __name__ == "__main__":
    jobs = []
    for cls in ("Mature", "Immature"):
        for side in (3, 4):
            d = os.path.join(ROOT, cls, f"Side{side}")
            for f in sorted(os.listdir(d), key=lambda x: int(x) if x.isdigit() else 0):
                if f.isdigit(): jobs.append((cls, side, int(f), os.path.join(d, f)))
    print(len(jobs), "spot folders", flush=True)
    with Pool() as pool: res = pool.map(job, jobs, chunksize=2)
    meta = pd.DataFrame([m for m, _ in res if m]); meta.to_csv(os.path.join(OUT, "spots_meta_s34.csv"), index=False)
    cols = ["cls", "side", "fid", "ch", "angle"] + [f"ring{i}" for i in range(NR)] + ["box75_R", "box75_raw"]
    df = pd.DataFrame([r for _, rs in res for r in rs], columns=cols); df.to_csv(os.path.join(OUT, "ring_long_s34.csv"), index=False)
    print(meta.groupby(["cls", "side"]).trunc.agg(["mean", "sum", "count"]).round(2)); print(df.shape, df.isna().sum().sum(), flush=True)


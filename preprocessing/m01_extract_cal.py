"""
Malwatte_01 – Step 1: intensity extraction
==========================================
Layout: Malwatte_01/<Class>/Side<1-4>/<fruit_id>/<x>_<angle>_<spot>_<ts>.png
  Class  : Immature | Mature   (ground truth = folder)
  x      : r=650nm g=520nm b=450nm, 0=dark frame
  angle  : 5deg steps (72) or 10deg steps (36) depending on session
Fruit IDs repeat across classes -> unique key = (Class, id).

Same ROI method as the earlier study: drop bottom 15 %, blur, brightest spot of the
angle-0 frame of that channel -> 75x75 crop reused for all angles.
Also records the mean of the dark frame (x=0) at the same angle in the same crop.

Output: analysis_output_m01/intensities_long.csv
  cls,side,fid,ch,angle,raw,dark
"""
import os, glob, cv2, numpy as np, pandas as pd
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "Malwatte_01")
OUT = os.path.join(BASE, "analysis_output_m01")
REF = {}
for fp in glob.glob(os.path.join(BASE, "Ref", "*.png")):
    p = os.path.basename(fp).split("_")
    REF[(p[0], int(p[1]))] = fp
REFG = {}  # lazy gray cache
os.makedirs(OUT, exist_ok=True)
CROP = 75


def gray(fp):
    img = cv2.imread(fp)
    if img is None:
        return None
    h = img.shape[0]
    return cv2.cvtColor(img[:int(h * 0.85), :], cv2.COLOR_BGR2GRAY).astype(np.float32)


def bright_box(g):
    b = cv2.GaussianBlur(g, (9, 9), 0)
    _, _, _, (bx, by) = cv2.minMaxLoc(b)
    h, w = g.shape
    half = CROP // 2
    l = max(0, bx - half); r = min(w, l + CROP)
    t = max(0, by - half); bt = min(h, t + CROP)
    return l, t, r, bt


def refgray(k):
    if k not in REFG:
        REFG[k] = gray(REF[k])
    return REFG[k]


def job(args):
    cls, side, fid, folder = args
    files = {}
    for fp in glob.glob(os.path.join(folder, "*.png")):
        p = os.path.basename(fp).split("_")
        if len(p) < 4:
            continue
        files[(p[0], int(p[1]))] = fp
    rows = []
    for ch in "rgb":
        angs = sorted(a for (c, a) in files if c == ch)
        if not angs:
            continue
        g0 = gray(files[(ch, angs[0])]) if (ch, 0) not in files else gray(files[(ch, 0)])
        box = bright_box(g0)
        l, t, r, b = box
        for a in angs:
            g = gray(files[(ch, a)])
            raw = float(g[t:b, l:r].mean()) if g is not None else np.nan
            dk = np.nan
            if ("0", a) in files:
                gd = gray(files[("0", a)])
                if gd is not None:
                    dk = float(gd[t:b, l:r].mean())
            rf = rd = np.nan
            if (ch, a) in REF and ("0", a) in REF:
                rf = float(refgray((ch, a))[t:b, l:r].mean())
                rd = float(refgray(("0", a))[t:b, l:r].mean())
            rows.append((cls, side, fid, ch, a, raw, dk, rf, rd))
    return rows


if __name__ == "__main__":
    jobs = []
    for cls in ("Immature", "Mature"):
        for s in range(1, 5):
            d = os.path.join(ROOT, cls, f"Side{s}")
            for f in sorted(os.listdir(d), key=lambda x: int(x) if x.isdigit() else 0):
                if f.isdigit():
                    jobs.append((cls, s, int(f), os.path.join(d, f)))
    print(f"{len(jobs)} spot folders")
    with Pool() as pool:
        res = pool.map(job, jobs, chunksize=4)
    df = pd.DataFrame([r for rs in res for r in rs],
                      columns=["cls", "side", "fid", "ch", "angle", "raw", "dark", "ref", "ref_dark"])
    df.to_csv(os.path.join(OUT, "intensities_long_with_ref.csv"), index=False)
    cal = df.copy()
    cal["raw"] = (df.raw - df.dark.fillna(0)) / (df.ref - df.ref_dark)
    CO = os.path.join(BASE, "analysis_output_m01_cal"); os.makedirs(CO, exist_ok=True)
    cal[["cls","side","fid","ch","angle","raw","dark"]].to_csv(os.path.join(CO, "intensities_long.csv"), index=False)
    print(df.shape, df.isna().sum().to_dict())


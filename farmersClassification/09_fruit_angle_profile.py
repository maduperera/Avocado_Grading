import os, glob, warnings, numpy as np, pandas as pd, cv2
from scipy import stats
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

# Load both uncalibrated (dark-subtracted fruit intensity) and Ref-calibrated reflectance
I_raw = pd.read_csv(os.path.join(BASE, "analysis_output_m01", "intensities_long.csv"))
I_cal = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "intensities_long.csv"))

for label, I in [("RAW DARK-SUBTRACTED AVOCADO", I_raw), ("REF-CALIBRATED AVOCADO", I_cal)]:
    n = I.groupby(["cls", "side", "fid", "ch"]).angle.transform("count")
    I5 = I[(n == 72) & I.side.isin([2, 3, 4]) & (I.raw > 0)].copy()
    
    ANG = np.arange(0, 360, 5.0)
    t = np.deg2rad(ANG)
    A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]
    
    rows = []
    for (cls, side, fid, ch), g in I5.groupby(["cls", "side", "fid", "ch"]):
        v = g.sort_values("angle").raw.values
        co, *_ = np.linalg.lstsq(A, v, rcond=None)
        fit = A @ co
        rel = v / v.mean()
        res = (v - fit) / co[2]
        for a, rv, rsv in zip(ANG.astype(int), rel, res):
            rows.append((cls, side, fid, ch, a, rv, rsv, abs(rsv)))
    df = pd.DataFrame(rows, columns=["cls", "side", "fid", "ch", "angle", "rel", "res", "abs_res"])
    df["fruit"] = df.cls + "_" + df.fid.astype(str)
    
    print(f"\n================ {label} (All 174 5-deg avocados, Sides 2-4) ================")
    for ch in ["r", "g", "b"]:
        sub = df[df.ch == ch]
        agg = sub.groupby("angle").agg(
            rel_mean=("rel", "mean"),
            rel_sd=("rel", "std"),
            res_mean=("res", "mean"),
            abs_res_mean=("abs_res", "mean"),
        )
        # Compute fruit-level ICC across sides 2,3,4 for rel and res at each angle
        iccs_rel = {}
        iccs_res = {}
        for a, ga in sub.groupby("angle"):
            for col, d_icc in [("rel", iccs_rel), ("res", iccs_res)]:
                p = ga.pivot_table(index="fruit", columns="side", values=col).dropna()
                x = p.values
                nn, k = x.shape
                msb = k * ((x.mean(1) - x.mean()) ** 2).sum() / (nn - 1)
                msw = ((x - x.mean(1)[:, None]) ** 2).sum() / (nn * (k - 1))
                d_icc[a] = (msb - msw) / (msb + (k - 1) * msw)
        agg["icc_rel"] = pd.Series(iccs_rel)
        agg["icc_res"] = pd.Series(iccs_res)
        
        top_bright = agg["rel_mean"].nlargest(5).index.tolist()
        top_dark = agg["rel_mean"].nsmallest(5).index.tolist()
        top_spike = agg["abs_res_mean"].nlargest(8).index.tolist()
        top_icc_rel = agg["icc_rel"].nlargest(5).index.tolist()
        
        print(f"\nChannel {ch.upper()}:")
        print(f"  Brightest angles (highest I(theta)/mean): {top_bright} (50 deg = {agg.loc[50,'rel_mean']:.4f}, rank {agg['rel_mean'].rank(ascending=False).loc[50]:.0f}/72)")
        print(f"  Darkest angles (lowest I(theta)/mean):    {top_dark} (135 deg = {agg.loc[135,'rel_mean']:.4f} rank {agg['rel_mean'].rank(ascending=True).loc[135]:.0f}/72; 140 deg = {agg.loc[140,'rel_mean']:.4f} rank {agg['rel_mean'].rank(ascending=True).loc[140]:.0f}/72)")
        print(f"  Largest |residual| spike angles:          {top_spike}")
        print(f"  Values at [50, 135, 140, 200, 275]:")
        for a in [50, 135, 140, 200, 275]:
            r_row = agg.loc[a]
            print(f"    {a:3d} deg: rel={r_row['rel_mean']:.4f} (±{r_row['rel_sd']:.4f}), res={r_row['res_mean']:+.4f}, |res|={r_row['abs_res_mean']:.4f} (rank {agg['abs_res_mean'].rank(ascending=False).loc[a]:.0f}/72), fruit-ICC(rel)={r_row['icc_rel']:.3f}, fruit-ICC(res)={r_row['icc_res']:.3f}")

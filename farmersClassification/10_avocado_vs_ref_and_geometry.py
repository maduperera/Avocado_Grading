import os, glob, warnings, numpy as np, pandas as pd, cv2
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

I_raw = pd.read_csv(os.path.join(BASE, "analysis_output_m01", "intensities_long.csv"))
I_cal = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "intensities_long.csv"))
ANG = np.arange(0, 360, 5.0)
t = np.deg2rad(ANG)
A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]

# Get Ref curve
ref_rel = {}
for ch in "rgb":
    v = np.array([cv2.imread(glob.glob(os.path.join(BASE, "Ref", f"{ch}_{int(a)}_0_*.png"))[0], 0)[:204].astype(float).mean() for a in ANG])
    ref_rel[ch] = v / v.mean()

for ch in ["r", "g", "b"]:
    n = I_cal.groupby(["cls", "side", "fid", "ch"]).angle.transform("count")
    sub_cal = I_cal[(n == 72) & I_cal.side.isin([2, 3, 4]) & (I_cal.raw > 0) & (I_cal.ch == ch)].copy()
    sub_raw = I_raw[(n == 72) & I_raw.side.isin([2, 3, 4]) & (I_raw.raw > 0) & (I_raw.ch == ch)].copy()
    
    # Calculate per-spot relative intensity and residual
    rows = []
    for (cls, side, fid), g in sub_cal.groupby(["cls", "side", "fid"]):
        v = g.sort_values("angle").raw.values
        co, *_ = np.linalg.lstsq(A, v, rcond=None)
        fit = A @ co
        rel = v / v.mean()
        res = (v - fit) / co[2]
        for a, rv, rsv in zip(ANG.astype(int), rel, res):
            rows.append((cls, side, fid, a, np.log(v[a//5]), rv, rsv))
    df = pd.DataFrame(rows, columns=["cls", "side", "fid", "angle", "logR", "rel", "res"])
    df["fruit"] = df.cls + "_" + df.fid.astype(str)
    
    # Within-class fruit ICC (demeaned by class so class difference doesn't inflate fruit ICC)
    for col in ["logR", "rel", "res"]:
        df[col + "_wc"] = df[col] - df.groupby(["cls", "angle"])[col].transform("mean")
    
    icc_wc = {}
    for a, ga in df.groupby("angle"):
        row_d = {}
        for col in ["logR_wc", "rel_wc", "res_wc"]:
            p = ga.pivot_table(index="fruit", columns="side", values=col).dropna()
            x = p.values; nn, k = x.shape
            msb = k * ((x.mean(1) - x.mean()) ** 2).sum() / (nn - 1)
            msw = ((x - x.mean(1)[:, None]) ** 2).sum() / (nn * (k - 1))
            row_d[col] = (msb - msw) / (msb + (k - 1) * msw)
        icc_wc[a] = row_d
    icc_df = pd.DataFrame(icc_wc).T
    
    print(f"\n--- Channel {ch.upper()} ---")
    print("Top 5 angles for individual fruit repeatability (within-class ICC across 3 sides of the SAME fruit):")
    print("  logR:", icc_df["logR_wc"].nlargest(5).round(3).to_dict())
    print("  rel :", icc_df["rel_wc"].nlargest(5).round(3).to_dict())
    print("  res :", icc_df["res_wc"].nlargest(5).round(3).to_dict())
    print("Values at [50, 135, 140, 200, 275]:")
    print(icc_df.loc[[50, 135, 140, 200, 275]].round(3))

# Save full 72-angle table and plot showing why 50, 135, 140, 200 are physically unique on the avocado
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pd.read_csv(os.path.join(HERE, "spot_dates.csv"))
I_cal_d = I_cal.merge(D[["cls", "side", "fid", "date"]], on=["cls", "side", "fid"])
n = I_cal_d.groupby(["cls", "side", "fid", "ch"]).angle.transform("count")
I5 = I_cal_d[(n == 72) & I_cal_d.side.isin([2, 3, 4]) & (I_cal_d.raw > 0)].copy()

all_rows = []
for ch in ["r", "g", "b"]:
    rows = []
    for (cls, side, fid, date), g in I5[I5.ch == ch].groupby(["cls", "side", "fid", "date"]):
        v = g.sort_values("angle").raw.values
        co, *_ = np.linalg.lstsq(A, v, rcond=None)
        fit = A @ co
        for a, vv, rv, rsv in zip(ANG.astype(int), np.log(v), v/v.mean(), (v-fit)/co[2]):
            rows.append((cls, side, fid, date, a, vv, rv, rsv))
    df = pd.DataFrame(rows, columns=["cls", "side", "fid", "date", "angle", "logR", "rel", "res"])
    df["fruit"] = df.cls + "_" + df.fid.astype(str)
    for col in ["logR", "rel", "res"]:
        df[col + "_wd"] = df[col] - df.groupby(["date", "angle"])[col].transform("mean")
    for a, ga in df.groupby("angle"):
        iccs = {}
        for col in ["logR_wd", "rel_wd", "res_wd"]:
            p = ga.pivot_table(index="fruit", columns="side", values=col).dropna()
            x = p.values; nn, k = x.shape
            msb = k * ((x.mean(1) - x.mean()) ** 2).sum() / (nn - 1)
            msw = ((x - x.mean(1)[:, None]) ** 2).sum() / (nn * (k - 1))
            iccs[col] = (msb - msw) / (msb + (k - 1) * msw)
        all_rows.append(dict(
            ch=ch, angle=a,
            avo_rel_mean=ga["rel"].mean(),
            avo_rel_sd=ga["rel"].std(),
            ref_rel=ref_rel[ch][a//5],
            icc_same_fruit_logR=iccs["logR_wd"],
            icc_same_fruit_rel=iccs["rel_wd"],
            icc_same_fruit_res=iccs["res_wd"]
        ))

out_df = pd.DataFrame(all_rows)
out_df.to_csv(os.path.join(HERE, "results", "fruit_angle_uniqueness.csv"), index=False)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
colors = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
labels = {"r": "Red (650 nm)", "g": "Green (520 nm)", "b": "Blue (450 nm)"}

for ch in ["r", "g", "b"]:
    sub = out_df[out_df.ch == ch]
    axes[0].plot(sub.angle, sub.avo_rel_mean, color=colors[ch], lw=2, label=f"Avocado {labels[ch]}")
    axes[0].plot(sub.angle, sub.ref_rel, color=colors[ch], lw=1, ls="--", alpha=0.6)
    axes[1].plot(sub.angle, sub.icc_same_fruit_logR, color=colors[ch], lw=2, label=labels[ch])
    axes[2].plot(sub.angle, sub.icc_same_fruit_rel, color=colors[ch], lw=2, label=labels[ch])

for ax in axes:
    for a, lbl in [(50, "50° (Glare Peak)"), (135, "135°"), (140, "140° (Glare-Blocked)"), (200, "200° (Steepest Slope)")]:
        ax.axvline(a, color="k", ls=":", alpha=0.6)

axes[0].set_ylabel("Relative Brightness\nI(θ) / mean(I)")
axes[0].set_title("1. Avocado Brightness Curve vs. Flat Reference (Dashed): 50° is Bright Co-Polarized Peak; 135°–140° is Dark Cross-Polarized Window")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)

axes[1].set_ylabel("Same-Fruit ICC\n(log R across 3 sides)")
axes[1].set_title("2. Individual Fruit Brightness Consistency (Across 3 Sides of Same Fruit): Peak at 135°–140° Where Surface Geometry Glare is Blocked")

axes[2].set_ylabel("Same-Fruit ICC\n(Rel & Geometry at θ)")
axes[2].set_title("3. Individual Fruit Shape/Phase Uniqueness (Across 3 Sides of Same Fruit): Peak at 200° Inflection Point")
axes[2].set_xlabel("Polarizer Angle θ (degrees)")

plt.tight_layout()
plt.savefig(os.path.join(HERE, "results", "fig_fruit_angle_uniqueness.png"), dpi=200)
print("Saved results/fruit_angle_uniqueness.csv and results/fig_fruit_angle_uniqueness.png")

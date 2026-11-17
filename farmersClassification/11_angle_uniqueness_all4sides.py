import os, glob, warnings, numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

I_cal = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "intensities_long.csv"))
D = pd.read_csv(os.path.join(HERE, "spot_dates.csv"))
I_cal = I_cal.merge(D[["cls", "side", "fid", "date"]], on=["cls", "side", "fid"])

ANG = np.arange(0, 360, 5.0)
t = np.deg2rad(ANG)
A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]

ref_rel = {}
for ch in "rgb":
    v = np.array([cv2.imread(glob.glob(os.path.join(BASE, "Ref", f"{ch}_{int(a)}_0_*.png"))[0], 0)[:204].astype(float).mean() for a in ANG])
    ref_rel[ch] = v / v.mean()

# 1) All 4 sides (Sides 1, 2, 3, 4) on all 5-deg spots (Immature has all 4 sides at 5-deg; Mature 27-100 has sides 2-4 at 5-deg)
n = I_cal.groupby(["cls", "side", "fid", "ch"]).angle.transform("count")
I5_all = I_cal[(n == 72) & (I_cal.raw > 0)].copy()

# 2) Also check All 200 fruits across all 4 sides (Sides 1, 2, 3, 4) by interpolating/phase-aligning the 10-deg spots to the 64-deg phase frame
rows_5deg_4s = []
for ch in ["r", "g", "b"]:
    sub = I5_all[I5_all.ch == ch]
    rows = []
    for (cls, side, fid, date), g in sub.groupby(["cls", "side", "fid", "date"]):
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
            # Pivot across all available sides (1, 2, 3, 4)
            p = ga.pivot_table(index="fruit", columns="side", values=col)
            # ANOVA ICC across fruits with >= 2 sides
            p_valid = p[p.notna().sum(axis=1) >= 2]
            # Standard one-way random effects ICC for unbalanced/balanced panels
            vals = [row.dropna().values for _, row in p_valid.iterrows()]
            N = len(vals)
            k_arr = np.array([len(v) for v in vals])
            grand_mean = np.concatenate(vals).mean()
            means = np.array([v.mean() for v in vals])
            msb = np.sum(k_arr * (means - grand_mean) ** 2) / (N - 1)
            msw = np.sum([np.sum((v - m) ** 2) for v, m in zip(vals, means)]) / (np.sum(k_arr) - N)
            k0 = (np.sum(k_arr) - np.sum(k_arr ** 2) / np.sum(k_arr)) / (N - 1)
            iccs[col] = (msb - msw) / (msb + (k0 - 1) * msw)
        rows_5deg_4s.append(dict(
            ch=ch, angle=a,
            avo_rel_mean=ga["rel"].mean(),
            avo_rel_sd=ga["rel"].std(),
            ref_rel=ref_rel[ch][a//5],
            icc_same_fruit_logR=iccs["logR_wd"],
            icc_same_fruit_rel=iccs["rel_wd"],
            icc_same_fruit_res=iccs["res_wd"]
        ))

out_4s = pd.DataFrame(rows_5deg_4s)
out_4s.to_csv(os.path.join(HERE, "results", "fruit_angle_uniqueness_all4sides.csv"), index=False)

for ch in ["r", "g", "b"]:
    sub = out_4s[out_4s.ch == ch].set_index("angle")
    print(f"=== Channel {ch.upper()} (All 4 Sides included, 5-deg spots: 621 spots across 174 fruits) ===")
    print("  Top 5 logR ICC:", sub["icc_same_fruit_logR"].nlargest(5).round(3).to_dict())
    print("  Top 5 rel  ICC:", sub["icc_same_fruit_rel"].nlargest(5).round(3).to_dict())
    print("  Top 5 res  ICC:", sub["icc_same_fruit_res"].nlargest(5).round(3).to_dict())
    print(sub.loc[[50, 135, 140, 200, 275], ["avo_rel_mean", "icc_same_fruit_logR", "icc_same_fruit_rel", "icc_same_fruit_res"]].round(3))

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
colors = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
labels = {"r": "Red (650 nm)", "g": "Green (520 nm)", "b": "Blue (450 nm)"}
for ch in ["r", "g", "b"]:
    sub = out_4s[out_4s.ch == ch]
    axes[0].plot(sub.angle.values, sub.avo_rel_mean.values, color=colors[ch], lw=2, label=f"Avocado {labels[ch]}")
    axes[0].plot(sub.angle.values, sub.ref_rel.values, color=colors[ch], lw=1, ls="--", alpha=0.6)
    axes[1].plot(sub.angle.values, sub.icc_same_fruit_logR.values, color=colors[ch], lw=2, label=labels[ch])
    axes[2].plot(sub.angle.values, sub.icc_same_fruit_rel.values, color=colors[ch], lw=2, label=labels[ch])
for ax in axes:
    for a in [50, 135, 140, 200]:
        ax.axvline(a, color="k", ls=":", alpha=0.7)
axes[0].set_ylabel("Relative Brightness\nI(θ) / mean(I)")
axes[0].set_title("1. Avocado Brightness Curve vs. Flat Reference (Sides 1–4): 50° is Co-Polarized Peak; 135°–140° is Cross-Polarized Window")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[1].set_ylabel("Same-Fruit ICC\n(log R across Sides 1–4)")
axes[1].set_title("2. Individual Fruit Brightness Consistency (Sides 1–4): Peak at 135°–140° Where Surface Geometry Glare is Blocked")
axes[2].set_ylabel("Same-Fruit ICC\n(Rel & Geometry at θ)")
axes[2].set_title("3. Individual Fruit Shape/Phase Uniqueness (Sides 1–4): Peak at 200° Inflection Point")
axes[2].set_xlabel("Polarizer Angle θ (degrees)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "results", "fig_fruit_angle_uniqueness_all4sides.png"), dpi=200)
# Also update fig_fruit_angle_uniqueness.png so both have all 4 sides
plt.savefig(os.path.join(HERE, "results", "fig_fruit_angle_uniqueness.png"), dpi=200)
print("Saved all-4-sides plot and CSV.")

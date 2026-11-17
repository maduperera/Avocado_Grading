"""
Step 5 - Is any analyzer angle (e.g. 50, 135, 140) special?
Clean subset: only 5-degree-step spots (all share the same polarizer zero point, fitted phase ~65 deg) and only Sides 2-4
(batch A: fruit 27-100, batch B: all) -> no phase alignment needed, 72 true angles incl. 135.
Fruit level (mean over the fruit's sides 2-4), calibrated relative reflectance (legacy extraction, analysis_output_m01_cal).
Per channel and angle:  log R(theta)  |  relative R(theta)/mean_theta  |  residual of the 2-theta fit (R - fit)/c
For each: Cohen's d (ready-now minus later) and AUC between the farmer classes, and within-batch r with DM.
Also shows the expected pattern if the curve is a pure sinusoid: rel(theta) = 1 + DoLP*cos(2(theta - phase)).
"""
import os, warnings, numpy as np, pandas as pd, openpyxl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = os.path.join(HERE, "results")
I = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "intensities_long.csv"))
n = I.groupby(["cls", "side", "fid", "ch"]).angle.transform("count"); I = I[(n == 72) & I.side.isin([2, 3, 4]) & (I.raw > 0)]
ws = openpyxl.load_workbook(os.path.join(BASE, "Avocado State.xlsx"), data_only=True).active
L = []
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is None: continue
    v = [x for x in (r[7], r[8]) if x is not None and 15 <= x <= 50]
    if v: L.append(("Mature" if r[0] <= 100 else "Immature", r[0] if r[0] <= 100 else r[0] - 100, np.mean(v)))
DM = pd.DataFrame(L, columns=["cls", "fid", "dm"])
ANG = list(range(0, 360, 5)); rows = []; phases = []
for (cls, side, fid, ch), g in I.groupby(["cls", "side", "fid", "ch"]):
    g = g.sort_values("angle"); a = g.angle.values.astype(float); v = g.raw.values; t = np.deg2rad(a)
    A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]; (p, q, c), *_ = np.linalg.lstsq(A, v, rcond=None); fit = A @ np.array([p, q, c])
    phases.append(np.degrees(0.5 * np.arctan2(q, p)) % 180)
    for ang, vv, ff in zip(a.astype(int), v, fit): rows.append((cls, side, fid, ch, ang, np.log(vv), vv / v.mean(), (vv - ff) / c))
S = pd.DataFrame(rows, columns=["cls", "side", "fid", "ch", "angle", "logR", "rel", "resid"])
Fr = S.groupby(["cls", "fid", "ch", "angle"])[["logR", "rel", "resid"]].mean().reset_index().merge(DM, on=["cls", "fid"], how="left")
Fr["y"] = (Fr.cls == "Mature").astype(int)
print("spots used:", S.drop_duplicates(["cls", "side", "fid"]).groupby("cls").size().to_dict(), "| fruit:", Fr.drop_duplicates(["cls", "fid"]).groupby("cls").size().to_dict(),
      "| fitted phase median %.1f deg (IQR %.1f-%.1f)" % (np.median(phases), *np.percentile(phases, [25, 75])))


def dval(x, y):
    a, b = x[y == 1], x[y == 0]; s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)); return (a.mean() - b.mean()) / s


res = []
for (ch, ang), g in Fr.groupby(["ch", "angle"]):
    for k in ("logR", "rel", "resid"):
        x, y = g[k].values, g.y.values; auc = roc_auc_score(y, x); p = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
        h = g.dropna(subset=["dm"]); xc = h[k] - h.groupby("cls")[k].transform("mean"); dc = h.dm - h.groupby("cls").dm.transform("mean"); r, pr = stats.pearsonr(xc, dc)
        res.append((ch, ang, k, dval(x, y), max(auc, 1 - auc), p, r, pr))
R = pd.DataFrame(res, columns=["ch", "angle", "quantity", "cohen_d", "AUC", "p_class", "r_DM_within", "p_DM"]); R.to_csv(os.path.join(OUT, "angle_significance.csv"), index=False)
ntest = len(R[R.quantity == "rel"]); bonf = 0.05 / ntest
pd.set_option("display.width", 200)
print(f"\nBonferroni threshold per quantity ({ntest} tests): p < {bonf:.1e}")
for k, lab in (("logR", "log reflectance"), ("rel", "relative intensity R/mean"), ("resid", "residual of 2-theta fit")):
    d = R[R.quantity == k]
    print(f"\n=== {lab}: farmer classes (fruit level) ===")
    for ch in "rgb":
        e = d[d.ch == ch].set_index("angle"); best = e.AUC.idxmax(); worst = e.AUC.idxmin()
        named = {a: (round(e.loc[a, 'cohen_d'], 2), round(e.loc[a, 'AUC'], 2)) for a in (50, 135, 140) if a in e.index}
        print(f"  {ch}: best angle {best} (d={e.loc[best,'cohen_d']:.2f}, AUC={e.loc[best,'AUC']:.2f}) | worst {worst} (AUC={e.loc[worst,'AUC']:.2f}) | range AUC {e.AUC.min():.2f}-{e.AUC.max():.2f} | 50/135/140 (d, AUC): {named} | n sig: {(e.p_class<bonf).sum()}/72")
    print(f"  within-batch r with DM: max |r| = {d.r_DM_within.abs().max():.2f} at {d.loc[d.r_DM_within.abs().idxmax(),['ch','angle']].tolist()} ; Bonferroni-sig: {(d.p_DM<bonf).sum()}")
print("\nNamed angles in detail (relative intensity):")
print(R[(R.quantity == "rel") & R.angle.isin([20, 50, 65, 110, 135, 140, 155])].pivot_table(index="angle", columns="ch", values=["cohen_d", "AUC", "r_DM_within"]).round(2).to_string())

# figure
ph = np.median(phases); fig, axs = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True); col = {"r": "#c0392b", "g": "#27ae60", "b": "#2980b9"}
for ax, (k, lab) in zip(axs, (("logR", "log reflectance"), ("rel", "relative intensity"), ("resid", "residual of 2θ fit"))):
    for ch in "rgb":
        e = R[(R.quantity == k) & (R.ch == ch)].sort_values("angle"); ax.plot(e.angle.values, e.cohen_d.values, "-", c=col[ch], label=dict(r="650 nm", g="520 nm", b="450 nm")[ch])
    for a in (50, 135, 140): ax.axvline(a, c="gray", ls=":", lw=1)
    for a in (ph, ph + 180): ax.axvline(a, c="k", ls="--", lw=.7)
    for a in (ph + 90, ph + 270): ax.axvline(a % 360, c="k", ls="-.", lw=.7)
    ax.axhline(0, c="k", lw=.4); ax.set_ylabel(f"Cohen's d\n({lab})"); ax.set_title(f"{lab}: ready-now minus later (fruit level)", fontsize=9)
axs[0].legend(fontsize=7, frameon=False); axs[-1].set_xlabel("analyzer angle (deg)   [dotted: 50/135/140; dashed: curve peak; dash-dot: curve trough]")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_angle_significance.png"), dpi=150); plt.close()


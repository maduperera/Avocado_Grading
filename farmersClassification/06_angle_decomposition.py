"""Which analyzer angles are significant, and why? Decompose every angle's relative intensity into
  fitted 2-theta part (= 1 + DoLP*cos2(theta-phase): carries only DoLP and phase)  +  residual (non-sinusoidal, instrument).
Data: 5-deg-step spots, sides 2-4, calibrated (same subset as 05_angle_significance.py). Fruit level.
Tests: farmer class (ready-now vs later; AUC, Mann-Whitney) and dry matter within batch (Pearson r)."""
import os, warnings, numpy as np, pandas as pd, openpyxl
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = os.path.join(HERE, "results")
I = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "intensities_long.csv"))
n = I.groupby(["cls", "side", "fid", "ch"]).angle.transform("count"); I = I[(n == 72) & I.side.isin([2, 3, 4]) & (I.raw > 0)]
ws = openpyxl.load_workbook(os.path.join(BASE, "Avocado State.xlsx"), data_only=True).active
L = [("Mature" if r[0] <= 100 else "Immature", r[0] if r[0] <= 100 else r[0] - 100, np.mean([x for x in (r[7], r[8]) if x is not None and 15 <= x <= 50]))
     for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None and any(x is not None and 15 <= x <= 50 for x in (r[7], r[8]))]
DM = pd.DataFrame(L, columns=["cls", "fid", "dm"])
rows, sp = [], []
for (cls, side, fid, ch), g in I.groupby(["cls", "side", "fid", "ch"]):
    g = g.sort_values("angle"); a = g.angle.values.astype(float); v = g.raw.values; t = np.deg2rad(a)
    A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]; (p, q, c), *_ = np.linalg.lstsq(A, v, rcond=None); fit = A @ np.array([p, q, c]); m = v.mean()
    sp.append((cls, fid, ch, np.hypot(p, q) / c, np.degrees(0.5 * np.arctan2(q, p)) % 180))
    for ang, vv, ff in zip(a.astype(int), v, fit): rows.append((cls, fid, ch, ang, vv / m, ff / m, (vv - ff) / m))
S = pd.DataFrame(rows, columns=["cls", "fid", "ch", "angle", "measured", "fit_part", "residual"])
Fr = S.groupby(["cls", "fid", "ch", "angle"]).mean().reset_index().merge(DM, on=["cls", "fid"], how="left"); Fr["y"] = (Fr.cls == "Mature").astype(int)
SP = pd.DataFrame(sp, columns=["cls", "fid", "ch", "DoLP", "phase"]).groupby(["cls", "fid", "ch"]).mean().reset_index().merge(DM, on=["cls", "fid"], how="left"); SP["y"] = (SP.cls == "Mature").astype(int)
def wr(x, d):
    h = d.dropna(subset=["dm"]); xc = h[x] - h.groupby("cls")[x].transform("mean"); dc = h.dm - h.groupby("cls").dm.transform("mean"); return stats.pearsonr(xc, dc)
res = []
for (ch, ang), g in Fr.groupby(["ch", "angle"]):
    for k in ("measured", "fit_part", "residual"):
        auc = roc_auc_score(g.y, g[k]); p = stats.mannwhitneyu(g[g.y == 1][k], g[g.y == 0][k]).pvalue; r, pr = wr(k, g)
        res.append((ch, ang, k, max(auc, 1 - auc), p, r, pr))
R = pd.DataFrame(res, columns=["ch", "angle", "part", "AUC_farmer", "p_farmer", "r_DM", "p_DM"]); R.to_csv(os.path.join(OUT, "angle_decomposition.csv"), index=False)
bonf = 0.05 / (72 * 3)
print(f"Bonferroni threshold (216 tests per part): p < {bonf:.1e}")
print("\nReference: whole-curve features (fruit level)")
for ch in "rgb":
    s = SP[SP.ch == ch]; a = roc_auc_score(s.y, s.DoLP); r, p = wr("DoLP", s); ph = s.phase.median()
    print(f"  {ch}: DoLP AUC_farmer={max(a,1-a):.2f}  r_DM={r:+.2f} (p={p:.3f}) | curve peak (phase) median {ph:.0f} deg -> trough {ph+90:.0f} deg, mid-points {ph-45:.0f}/{ph+45:.0f} deg")
for ch in "rgb":
    print(f"\n=== {dict(r='650 nm', g='520 nm', b='450 nm')[ch]} ===")
    d = R[R.ch == ch]
    for k in ("measured", "fit_part", "residual"):
        e = d[d.part == k]; tf = e.sort_values("AUC_farmer", ascending=False).head(4); td = e.reindex(e.r_DM.abs().sort_values(ascending=False).index).head(4)
        print(f"  {k:9s} farmer: sig {int((e.p_farmer<bonf).sum()):2d}/72, top {[(int(a), round(v,2)) for a, v in zip(tf.angle, tf.AUC_farmer)]}"
              f" | DM: sig {int((e.p_DM<bonf).sum())}/72, top {[(int(a), round(v,2)) for a, v in zip(td.angle, td.r_DM)]}")
# figure
fig, axs = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True); e = R[R.ch == "r"].sort_values("angle")
for k, c in (("measured", "k"), ("fit_part", "#2a9d8f"), ("residual", "#e76f51")):
    s = e[e.part == k]; axs[0].plot(s.angle.values, s.AUC_farmer.values, c=c, label=k); axs[1].plot(s.angle.values, s.r_DM.values, c=c, label=k)
ph = SP[SP.ch == "r"].phase.median()
for ax in axs:
    for a0 in (ph, ph + 180): ax.axvline(a0, c="gray", ls="--", lw=.7)
    for a0 in (ph + 90, ph + 270): ax.axvline(a0 % 360, c="gray", ls="-.", lw=.7)
axs[0].set_ylabel("AUC, ready-now vs later"); axs[1].set_ylabel("within-batch r with DM"); axs[1].axhline(0, c="k", lw=.4); axs[0].legend(fontsize=7, frameon=False)
axs[1].set_xlabel("analyzer angle (deg)   [dashed: curve peak, dash-dot: curve trough]"); axs[0].set_title("650 nm: measured relative intensity = fitted (DoLP, phase) part + residual", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_angle_decomposition.png"), dpi=150)


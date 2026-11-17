"""
Step 3 - Same classification, but ALL spots reduced to the common 10-degree analyzer grid (5-degree sessions sub-sampled to multiples of 10).
Shared-spot ring features are re-fitted on that grid only and compared with the native-grid fit (5 or 10 deg steps).
Also: (a) fidelity of the 10-degree-grid features vs the native ones, (b) can the features tell 10-deg-step sessions from 5-deg-step
sessions WITHIN the Mature folder (same sides, fruit 1-26 vs 27-100)?, (c) probes kept for reference.
Base code untouched; reads analysis_output_m01_v2/{ring_long.csv, ring_long_s34.csv, spots_meta*.csv}.
"""
import os, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); V2 = os.path.join(BASE, "analysis_output_m01_v2"); OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)

R = pd.concat([pd.read_csv(os.path.join(V2, "ring_long.csv")), pd.read_csv(os.path.join(V2, "ring_long_s34.csv"))])
M = pd.concat([pd.read_csv(os.path.join(V2, "spots_meta.csv")), pd.read_csv(os.path.join(V2, "spots_meta_s34.csv"))])
R = R.merge(M[["cls", "side", "fid", "trunc", "nang"]], on=["cls", "side", "fid"]); R = R[R.trunc == 0]
R["step"] = np.where(R.nang == 36, 10, 5)
RINGS = [f"ring{i}" for i in range(6)]


def fit(a, v):
    t = np.deg2rad(a); A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]; (p, q, c), *_ = np.linalg.lstsq(A, v, rcond=None)
    return (np.hypot(p, q) / c, c, np.degrees(0.5 * np.arctan2(q, p)) % 180) if c > 0 else (np.nan,) * 3


def features(df, grid10):
    d = df[df.angle % 10 == 0] if grid10 else df
    out = {}
    for ring in RINGS:
        P = d.pivot_table(index=["cls", "side", "fid"], columns=["ch", "angle"], values=ring); idx = P.index
        for ch in "rgb":
            sub = P[ch]; angs = np.array(sub.columns, float); res = []
            for row in sub.values:
                m = ~np.isnan(row); res.append(fit(angs[m], row[m]) if m.sum() > 12 else (np.nan,) * 3)
            o = np.array(res); out[f"{ring}_{ch}_DoLP"] = o[:, 0]; out[f"{ring}_{ch}_logR"] = np.log(o[:, 1])
            if ring == "ring0": out[f"phase_{ch}_c2"] = np.cos(2 * np.deg2rad(o[:, 2])); out[f"phase_{ch}_s2"] = np.sin(2 * np.deg2rad(o[:, 2]))
    X = pd.DataFrame(out, index=idx).reset_index(); X["y"] = (X.cls == "Mature").astype(int); X["fruit"] = X.cls.str[0] + X.fid.astype(str)
    st = df.drop_duplicates(["cls", "side", "fid"])[["cls", "side", "fid", "step"]]; return X.merge(st, on=["cls", "side", "fid"])


NAT = features(R, False); G10 = features(R, True)
print("spots:", len(NAT), "(non-truncated, sides 1-4)  grid10 == native for 10-deg sessions; 5-deg sessions subsampled")
dcols = [f"{r}_{c}_DoLP" for r in RINGS for c in "rgb"]; rcols = [f"{r}_{c}_logR" for r in RINGS for c in "rgb"]; pcols = [c for c in NAT.columns if c.startswith("phase_")]
sets = {"DoLP only": dcols, "Reflectance only": rcols, "DoLP + reflectance": dcols + rcols, "Red DoLP only": [c for c in dcols if "_r_" in c], "PROBE phase only": pcols, "PROBE step size only": ["step"]}

# ---- (a) fidelity
print("\n(a) feature fidelity, 10-deg grid vs native fit (spot level, Pearson r):")
for f in ["ring0_r_DoLP", "ring0_g_DoLP", "ring5_r_DoLP", "ring0_g_logR", "ring5_g_logR"]:
    m = NAT.merge(G10, on=["cls", "side", "fid"], suffixes=("_n", "_10")); r5 = m[m.step_n == 5]
    print(f"  {f}: all r={np.corrcoef(m[f+'_n'], m[f+'_10'])[0, 1]:.4f} | 5-deg spots r={np.corrcoef(r5[f+'_n'], r5[f+'_10'])[0, 1]:.4f} | mean |diff| DoLP/logR = {np.mean(np.abs(m[f+'_n'] - m[f+'_10'])):.4f}")

models = {"LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
          "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(C=1, probability=True, random_state=0)),
          "RandForest": lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=-1)}


def cv_fruit(T, cols, reps=5):
    G = T.groupby("fruit").agg({**{c: "mean" for c in cols}, "y": "first"}).dropna(); X, y = G[cols].values, G.y.values; res = {}
    for mn, mk in models.items():
        b, a = [], []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=0).split(X, y):
            m = mk().fit(X[tr], y[tr]); b.append(balanced_accuracy_score(y[te], m.predict(X[te]))); a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        res[mn] = (np.mean(b), np.std(b), np.mean(a), len(y))
    return res


def cv_spot(T, cols, reps=2, label="y"):
    G = T.dropna(subset=cols).reset_index(drop=True); X, y, g = G[cols].values, G[label].values, G.fruit.values; res = {}
    for mn, mk in models.items():
        b, a = [], []
        for r in range(reps):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
                m = mk().fit(X[tr], y[tr]); b.append(balanced_accuracy_score(y[te], m.predict(X[te]))); a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        res[mn] = (np.mean(b), np.std(b), np.mean(a), len(y))
    return res


rows = []
for grid, T in (("native (5 or 10 deg)", NAT), ("10-deg grid only", G10)):
    for level, fn in (("fruit", cv_fruit), ("spot", cv_spot)):
        for name, cols in sets.items():
            r = fn(T, cols); mn = max(r, key=lambda k: r[k][0]); rows.append((grid, level, name, mn, *r[mn]))
    print("done", grid, flush=True)
Rdf = pd.DataFrame(rows, columns=["grid", "level", "feature_set", "model", "balanced_acc", "sd", "AUC", "n"]); Rdf.to_csv(os.path.join(OUT, "classification_10deg_vs_native.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== Mature vs Immature (folder labels), best model; chance 0.50 ===")
print(Rdf.pivot_table(index=["level", "feature_set"], columns="grid", values="balanced_acc").round(3).to_string())
print("\nAUC:"); print(Rdf.pivot_table(index=["level", "feature_set"], columns="grid", values="AUC").round(3).to_string())

# ---- (b) within Mature: 10-deg-step sessions vs 5-deg-step sessions, same sides (2-4), fruit 1-26 vs 27-100
print("\n(b) WITHIN the Mature folder, Sides 2-4: can features tell 10-deg-step sessions (fruit 1-26) from 5-deg-step sessions (fruit 27-100)?")
W = G10[(G10.cls == "Mature") & (G10.side.isin([2, 3, 4]))].copy(); W["ystep"] = (W.step == 10).astype(int)
print("   spots:", W.ystep.value_counts().to_dict(), "| fruit:", W.groupby('ystep').fruit.nunique().to_dict())
bb = []
for name in ("DoLP only", "Reflectance only", "DoLP + reflectance", "PROBE phase only"):
    r = cv_spot(W.assign(y=W.ystep), sets[name], reps=3); mn = max(r, key=lambda k: r[k][0]); bb.append((name, mn, r[mn][0], r[mn][2])); print(f"   {name:22s} {mn:10s} bal.acc={r[mn][0]:.3f}  AUC={r[mn][2]:.3f}")

# ---- figure
fig, ax = plt.subplots(figsize=(8, 4.6)); d = Rdf[Rdf.level == "fruit"]; names = list(sets); x = np.arange(len(names)); w = .38
for i, (grid, c) in enumerate([("native (5 or 10 deg)", "#adb5bd"), ("10-deg grid only", "#2a9d8f")]):
    v = [d[(d.grid == grid) & (d.feature_set == n)].balanced_acc.iloc[0] for n in names]; ax.bar(x + (i - .5) * w, v, w, color=c, label=grid)
ax.axhline(.5, c="k", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7); ax.set_ylim(.4, 1.03); ax.set_ylabel("balanced accuracy (fruit level)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Folder-label classification: native vs common 10-deg analyzer grid\n(PROBE = no fruit information)", fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_10deg_vs_native.png"), dpi=160); plt.close()


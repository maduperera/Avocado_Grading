"""Anisotropy-type features vs the FARMER classification (no dry matter).
Data: 5-deg spots, sides 2-4, calibrated (ready-now = 'Mature' batch, Oct 30; later = 'Immature' batch, Nov 1 & 2).
Per spot and channel: DoLP, orientation (phase, as cos2phi/sin2phi), residual after the 2-theta fit (72 values).
Fruit level (mean over sides), repeated stratified 5-fold CV, best of LogReg/RBF-SVM (+PCA for the residual).
Comparison that controls for 'day': can the SAME features separate the two capture days INSIDE the later class (Nov 1 vs Nov 2)?"""
import os, warnings, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
I = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "intensities_long.csv"))
n = I.groupby(["cls", "side", "fid", "ch"]).angle.transform("count"); I = I[(n == 72) & I.side.isin([2, 3, 4]) & (I.raw > 0)]
D = pd.read_csv(os.path.join(HERE, "spot_dates.csv")); I = I.merge(D[["cls", "side", "fid", "date"]], on=["cls", "side", "fid"])
ANG = np.arange(0, 360, 5.0); t = np.deg2rad(ANG); A = np.c_[np.cos(2 * t), np.sin(2 * t), np.ones_like(t)]
rec = {}
for (cls, side, fid, ch, date), g in I.groupby(["cls", "side", "fid", "ch", "date"]):
    v = g.sort_values("angle").raw.values; co, *_ = np.linalg.lstsq(A, v, rcond=None); ph = 0.5 * np.arctan2(co[1], co[0])
    rec.setdefault((cls, fid, date), {}).setdefault(ch, []).append((np.hypot(co[0], co[1]) / co[2], np.cos(2 * ph), np.sin(2 * ph), (v - A @ co) / co[2]))
rows = []
for (cls, fid, date), chd in rec.items():
    r = {"cls": cls, "fid": fid, "date": date, "y": int(cls == "Mature")}
    for ch, L in chd.items():
        r[f"DoLP_{ch}"] = np.mean([x[0] for x in L]); r[f"c2_{ch}"] = np.mean([x[1] for x in L]); r[f"s2_{ch}"] = np.mean([x[2] for x in L])
        for k, val in enumerate(np.mean([x[3] for x in L], 0)): r[f"res{k*5}_{ch}"] = val
    rows.append(r)
F = pd.DataFrame(rows).dropna()
print("fruit:", F.groupby(["cls", "date"]).size().to_dict())
sets = {"orientation (phase) only": [f"{q}_{c}" for q in ("c2", "s2") for c in "rgb"],
        "DoLP only": [f"DoLP_{c}" for c in "rgb"],
        "DoLP + orientation (all a real anisotropy can change)": [f"{q}_{c}" for q in ("DoLP", "c2", "s2") for c in "rgb"],
        "residual (non-sinusoidal part), 3 colours": [c for c in F.columns if c.startswith("res")],
        "residual at 50/135/140/200/275 deg only": [f"res{a}_{c}" for a in (50, 135, 140, 200, 275) for c in "rgb"]}
def models(ncol):
    m = {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000)), "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=1, probability=True, random_state=0))}
    if ncol > 30: m["PCA10+LogReg"] = make_pipeline(StandardScaler(), PCA(10), LogisticRegression(C=0.3, max_iter=4000))
    return m
def cv(X, y):
    best = None
    for mn, mdl in models(X.shape[1]).items():
        b, a = [], []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y):
            mdl.fit(X[tr], y[tr]); b.append(balanced_accuracy_score(y[te], mdl.predict(X[te]))); a.append(roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1]))
        if best is None or np.mean(b) > best[1]: best = (mn, np.mean(b), np.mean(a))
    return best
later = F[F.cls == "Immature"].copy(); later["yd"] = (later.date == later.date.max()).astype(int)
out = []
for name, cols in sets.items():
    m1 = cv(F[cols].values, F.y.values); m2 = cv(later[cols].values, later.yd.values)
    out.append((name, len(cols), m1[0], m1[1], m1[2], m2[1], m2[2]))
O = pd.DataFrame(out, columns=["features", "n_feat", "model", "ready_vs_later_balacc", "ready_vs_later_AUC", "NovI_vs_NovII_balacc", "NovI_vs_NovII_AUC"])
pd.set_option("display.width", 220); print(O.round(3).to_string(index=False))
O.to_csv(os.path.join(HERE, "results", "anisotropy_vs_farmer.csv"), index=False)


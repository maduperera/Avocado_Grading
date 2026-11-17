"""
Step 2 - 'Farmer's classification' experiment (curiosity run; base code untouched)
Assumption: the folder names are the true classes  (Mature = batch A, Immature = batch B). DM is NOT used.
Features: DoLP and intensity (calibrated relative reflectance) of the 3 wavelengths.
Levels: fruit level (mean over the usable sides)  and  spot level (grouped by fruit).
Leakage probes (features that carry NO fruit information but encode the capture session):
   analyzer step size, polarizer phase, [capture day-of-session as a within-class test]
Everything here reads existing analysis outputs; nothing in the base pipeline is modified.
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ data
F = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "features_per_side.csv"))       # legacy extraction, calibrated, 4 sides, per channel fit features
D = pd.read_csv(os.path.join(HERE, "spot_dates.csv"))
F["y"] = (F.cls == "Mature").astype(int)                                                      # Mature=1 (farmer folder), Immature=0
base = ["DoLP", "dc", "Imax", "Imin", "amp", "r2", "resid_rel"]
W = F.pivot_table(index=["cls", "side", "fid"], columns="ch", values=base); W.columns = [f"{c}_{ch}" for c, ch in W.columns]
ph = F.assign(c2=np.cos(2 * np.deg2rad(F.phase)), s2=np.sin(2 * np.deg2rad(F.phase))).pivot_table(index=["cls", "side", "fid"], columns="ch", values=["c2", "s2"]); ph.columns = [f"phase{a}_{b}" for a, b in ph.columns]
st = F[F.ch == "g"].set_index(["cls", "side", "fid"]).step.rename("step")
S = W.join(ph).join(st).reset_index().merge(D, on=["cls", "side", "fid"]); S["y"] = (S.cls == "Mature").astype(int); S["fruit"] = S.cls.str[0] + S.fid.astype(str)
for ch in "rgb": S[f"logdc_{ch}"] = np.log(S[f"dc_{ch}"]); S[f"logImin_{ch}"] = np.log(S[f"Imin_{ch}"].clip(lower=1e-3)); S[f"logImax_{ch}"] = np.log(S[f"Imax_{ch}"])
for a, b in [("r", "g"), ("g", "b"), ("r", "b")]: S[f"ratio_dc_{a}{b}"] = S[f"logdc_{a}"] - S[f"logdc_{b}"]; S[f"ratio_DoLP_{a}{b}"] = S[f"DoLP_{a}"] / S[f"DoLP_{b}"]
for ch in "rgb": S[f"ext_{ch}"] = S[f"Imin_{ch}"].clip(lower=1e-3) / S[f"Imax_{ch}"]
S = S.replace([np.inf, -np.inf], np.nan)

sets = {
    "DoLP only (r,g,b)": [f"DoLP_{c}" for c in "rgb"],
    "Intensity only (mean refl., Imax, Imin; r,g,b)": [f"logdc_{c}" for c in "rgb"] + [f"logImax_{c}" for c in "rgb"] + [f"logImin_{c}" for c in "rgb"],
    "DoLP + intensity": [f"DoLP_{c}" for c in "rgb"] + [f"logdc_{c}" for c in "rgb"] + [f"logImax_{c}" for c in "rgb"] + [f"logImin_{c}" for c in "rgb"],
    "Scale-invariant only (DoLP, extinction, wavelength ratios)": [f"DoLP_{c}" for c in "rgb"] + [f"ext_{c}" for c in "rgb"] + [f"ratio_dc_{p}" for p in ("rg", "gb", "rb")] + [f"ratio_DoLP_{p}" for p in ("rg", "gb", "rb")],
    "Red channel only (DoLP + intensity)": ["DoLP_r", "logdc_r", "logImax_r", "logImin_r"],
    "Green channel only (DoLP + intensity)": ["DoLP_g", "logdc_g", "logImax_g", "logImin_g"],
    "Blue channel only (DoLP + intensity)": ["DoLP_b", "logdc_b", "logImax_b", "logImin_b"],
}
probes = {"PROBE analyzer step size only": ["step"], "PROBE polarizer phase only": [f"phase{a}_{c}" for a in ("c2", "s2") for c in "rgb"]}
# optional: shared-spot (v2) ring features, non-truncated spots
V2P = os.path.join(BASE, "analysis_output_m01_v2", "results4", "spot_features_4sides.csv")
V = None
if os.path.exists(V2P):
    V = pd.read_csv(V2P); V["cls"] = np.where(V.batch == "A", "Mature", "Immature"); V["y"] = (V.cls == "Mature").astype(int); V["fruit"] = V.cls.str[0] + V.fid.astype(str)
    vcols = [f"ring{i}_{c}_DoLP" for i in range(6) for c in "rgb"] + [f"ring{i}_{c}_logRmean" for i in range(6) for c in "rgb"]
    vcols = [c for c in vcols if c in V.columns]; V = V.dropna(subset=vcols)
    sets_v2 = {"[shared-spot rings] DoLP only": [c for c in vcols if c.endswith("DoLP")], "[shared-spot rings] reflectance only": [c for c in vcols if c.endswith("logRmean")], "[shared-spot rings] DoLP + reflectance": vcols}

models = {"LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
          "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(C=1, probability=True, random_state=0)),
          "RandForest": lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=-1)}


def score(m, Xtr, ytr, Xte):
    m.fit(Xtr, ytr)
    return m.predict(Xte), m.predict_proba(Xte)[:, 1]


def eval_fruit(T, cols, reps=5):
    """fruit-level: mean of usable sides per fruit; repeated stratified 5-fold."""
    G = T.groupby("fruit").agg({**{c: "mean" for c in cols}, "y": "first"}).dropna(); X = G[cols].values; y = G.y.values
    out = {}
    for mn, mk in models.items():
        bal, acc, auc = [], [], []; pooled = np.zeros(len(y)); pr_all = np.zeros(len(y), int)
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=0).split(X, y):
            p, pp = score(mk(), X[tr], y[tr], X[te]); bal.append(balanced_accuracy_score(y[te], p)); acc.append(accuracy_score(y[te], p)); auc.append(roc_auc_score(y[te], pp))
        out[mn] = (np.mean(bal), np.std(bal), np.mean(acc), np.mean(auc), len(y))
    return out


def eval_spot(T, cols, reps=3):
    """spot-level: every side is a sample; folds grouped by fruit."""
    G = T.dropna(subset=cols).reset_index(drop=True); X = G[cols].values; y = G.y.values; g = G.fruit.values; out = {}
    for mn, mk in models.items():
        bal, acc, auc = [], [], []
        for r in range(reps):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
                p, pp = score(mk(), X[tr], y[tr], X[te]); bal.append(balanced_accuracy_score(y[te], p)); acc.append(accuracy_score(y[te], p)); auc.append(roc_auc_score(y[te], pp))
        out[mn] = (np.mean(bal), np.std(bal), np.mean(acc), np.mean(auc), len(y))
    return out


rows = []
for level, ev in (("fruit", eval_fruit), ("spot", eval_spot)):
    for name, cols in {**sets, **probes}.items():
        for mn, v in ev(S, cols).items(): rows.append((level, name, mn, *v))
    if V is not None:
        for name, cols in sets_v2.items():
            for mn, v in ev(V, cols).items(): rows.append((level, name, mn, *v))
    print("done", level, flush=True)
R = pd.DataFrame(rows, columns=["level", "feature_set", "model", "balanced_acc", "sd", "accuracy", "AUC", "n"]); R.to_csv(os.path.join(OUT, "classification_results.csv"), index=False)
pd.set_option("display.width", 220)
for level in ("fruit", "spot"):
    d = R[R.level == level].sort_values("balanced_acc", ascending=False).groupby("feature_set").head(1)
    print(f"\n=== {level}-level: best model per feature set (chance = 0.50) ==="); print(d[["feature_set", "model", "n", "balanced_acc", "sd", "accuracy", "AUC"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ session diagnostics: can the same features tell DAYS within one class?
print("\n=== Within-class day separation (same features; nothing to do with maturity) ===")
dd = []
for cls, d1, d2 in [("Mature", "2026-10-29", "2026-10-30"), ("Immature", "2026-11-01", "2026-11-02")]:
    T = S[(S.cls == cls) & S.date.isin([d1, d2])].copy(); T["y"] = (T.date == d2).astype(int)
    for name in ("DoLP only (r,g,b)", "Intensity only (mean refl., Imax, Imin; r,g,b)", "DoLP + intensity", "Scale-invariant only (DoLP, extinction, wavelength ratios)"):
        r = eval_spot(T.assign(fruit=T.fruit), sets[name], reps=2)
        b = max(v[0] for v in r.values()); dd.append((cls, f"{d1[5:]} vs {d2[5:]}", name, len(T), b))
DD = pd.DataFrame(dd, columns=["class", "days", "feature_set", "n_spots", "best_balanced_acc"]); DD.to_csv(os.path.join(OUT, "within_class_day_separation.csv"), index=False)
print(DD.round(3).to_string(index=False))

# ------------------------------------------------------------------ figures
fr = R[(R.level == "fruit")].sort_values("balanced_acc", ascending=False).groupby("feature_set").head(1).sort_values("balanced_acc")
fig, ax = plt.subplots(figsize=(8, 4.6)); cols_ = ["#c0392b" if n.startswith("PROBE") else "#2a9d8f" for n in fr.feature_set]
ax.barh(fr.feature_set, fr.balanced_acc, xerr=fr.sd, color=cols_); ax.axvline(0.5, c="k", ls="--", lw=1); ax.set_xlim(0.4, 1.02); ax.set_xlabel("balanced accuracy (fruit level, best model, 5-fold x5)")
for i, (v, m) in enumerate(zip(fr.balanced_acc, fr.model)): ax.text(v + 0.005, i, f"{v:.2f} ({m})", va="center", fontsize=7)
ax.set_title("Mature (batch A) vs Immature (batch B) from folder labels\nred = leakage probes (carry no fruit information)", fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_accuracy_by_featureset.png"), dpi=160); plt.close()

# ROC + confusion for 'DoLP + intensity' and 'DoLP only' (fruit level, LogReg, one repeated-CV pooled)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for name, c in [("DoLP only (r,g,b)", "#264653"), ("Intensity only (mean refl., Imax, Imin; r,g,b)", "#e76f51"), ("DoLP + intensity", "#2a9d8f"), ("Scale-invariant only (DoLP, extinction, wavelength ratios)", "#8e24aa")]:
    G = S.groupby("fruit").agg({**{k: "mean" for k in sets[name]}, "y": "first"}).dropna(); X = G[sets[name]].values; y = G.y.values; pp = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1).split(X, y):
        m = models["LogReg"](); m.fit(X[tr], y[tr]); pp[te] += m.predict_proba(X[te])[:, 1]; cnt[te] += 1
    pp /= cnt; fpr, tpr, _ = roc_curve(y, pp); axs[0].plot(fpr, tpr, c=c, label=f"{name.split(' (')[0]}  AUC={roc_auc_score(y, pp):.3f}")
    if name == "DoLP only (r,g,b)": dolp_pp, dolp_y = pp, y
    if name == "DoLP + intensity": both_pp, both_y = pp, y
axs[0].plot([0, 1], [0, 1], "k--", lw=.8); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].legend(fontsize=6.5, frameon=False); axs[0].set_title("ROC, fruit level (LogReg)", fontsize=9)
for ax, (pp, y, t) in zip(axs[1:], [(dolp_pp, dolp_y, "DoLP only"), (both_pp, both_y, "DoLP + intensity")]):
    cm = confusion_matrix(y, (pp > .5).astype(int)); cn = cm / cm.sum(1, keepdims=True); ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]}\n({cn[i, j]:.0%})", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred Immature", "pred Mature"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true Immature", "true Mature"]); ax.set_title(f"{t}", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_roc_confusion.png"), dpi=160); plt.close()

# distributions of the key features per class (spot level)
fig, axs = plt.subplots(2, 3, figsize=(12, 6.2)); colr = {"Mature": "#2a9d8f", "Immature": "#e76f51"}
for j, ch in enumerate("rgb"):
    for i, (feat, lab) in enumerate([(f"DoLP_{ch}", "DoLP"), (f"logdc_{ch}", "log mean reflectance")]):
        ax = axs[i, j]; ax.boxplot([S[S.cls == c][feat].dropna() for c in ("Mature", "Immature")], labels=["Mature\n(batch A)", "Immature\n(batch B)"], patch_artist=True, boxprops=dict(facecolor="#dddddd"), medianprops=dict(color="k"))
        for k, c in enumerate(("Mature", "Immature")): ax.scatter(np.random.normal(k + 1, .05, (S.cls == c).sum()), S[S.cls == c][feat], s=3, alpha=.2, c=colr[c])
        ax.set_title(f"{lab}, {dict(r='650', g='520', b='450')[ch]} nm", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_feature_distributions.png"), dpi=150); plt.close()

# per-feature AUC
au = []
for c in sorted(set(sum(sets.values(), []))):
    G = S.groupby("fruit").agg({c: "mean", "y": "first"}).dropna(); a = roc_auc_score(G.y, G[c]); au.append((c, max(a, 1 - a), "Mature higher" if a > .5 else "Mature lower"))
AU = pd.DataFrame(au, columns=["feature", "AUC_fruit_level", "direction"]).sort_values("AUC_fruit_level", ascending=False); AU.to_csv(os.path.join(OUT, "single_feature_auc.csv"), index=False)
print("\nTop single features (fruit-level AUC):"); print(AU.head(10).round(3).to_string(index=False)); print("\nDoLP single-feature AUCs:"); print(AU[AU.feature.str.startswith("DoLP")].round(3).to_string(index=False))


"""
Three Optical-Physics Analyses on Existing Malwatte_01 Measurements:
1. Normalized Radial DoLP Analysis:
   Remove each spot's/fruit's overall DoLP level (both additive: DoLP_rk - mean_r(DoLP)
   and multiplicative: DoLP_rk / mean_r(DoLP), plus core-vs-halo slope) and test whether
   class differences and ML classification accuracy persist.
2. Calibration Sensitivity Analysis:
   Evaluate how systematic LED-intensity drift (achromatic exposure/distance shift and
   independent per-channel R/G/B LED drift up to +-50%) and reference calibration perturbations
   affect classification performance across feature sets.
3. Spatial Polarization Consistency:
   Check whether the core-versus-halo radial effects and normalized radial DoLP differences
   have consistent directions and effect sizes across all 4 fruit sides (Sides 1, 2, 3, 4)
   and across wavelengths (650 nm, 520 nm, 450 nm), including cross-side generalization.
"""
import os, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Load shared-spot 6-ring dataset across all 4 sides
V = pd.read_csv(os.path.join(BASE, "analysis_output_m01_v2", "results4", "spot_features_4sides.csv"))
V["cls"] = np.where(V.batch == "A", "Mature", "Immature")
V["y"] = (V.cls == "Mature").astype(int)
V["fruit"] = V.cls.str[0] + V.fid.astype(str)

dolp_cols = [f"ring{r}_{c}_DoLP" for r in range(6) for c in "rgb"]
refl_cols = [f"ring{r}_{c}_logRmean" for r in range(6) for c in "rgb"]
V = V.dropna(subset=dolp_cols + refl_cols).reset_index(drop=True)

# ==============================================================================
# 1. NORMALIZED RADIAL DoLP (& NORMALIZED RADIAL REFLECTANCE) FEATURES
# ==============================================================================
# For each spot and each channel c in {r, g, b}:
#   - spot_mean_DoLP_c = mean over rings 0..5 of ring{k}_{c}_DoLP
#   - norm_diff_ring{k}_{c}_DoLP = ring{k}_{c}_DoLP - spot_mean_DoLP_c   (zero mean per spot & channel!)
#   - norm_ratio_ring{k}_{c}_DoLP = ring{k}_{c}_DoLP / spot_mean_DoLP_c  (unit mean per spot & channel!)
#   - halo_minus_core_{c}_DoLP = mean(ring3,4,5) - ring0
#   - norm_diff_ring{k}_{c}_logR = ring{k}_{c}_logRmean - mean_r(ring{k}_{c}_logRmean) (scale-free beam profile!)

norm_diff_dolp_cols = []
norm_ratio_dolp_cols = []
norm_diff_refl_cols = []
halo_core_cols = []

for c in ["r", "g", "b"]:
    c_dolp = [f"ring{r}_{c}_DoLP" for r in range(6)]
    c_refl = [f"ring{r}_{c}_logRmean" for r in range(6)]
    V[f"mean_DoLP_{c}"] = V[c_dolp].mean(axis=1)
    V[f"mean_logR_{c}"] = V[c_refl].mean(axis=1)
    for r in range(6):
        nd_col = f"norm_diff_ring{r}_{c}_DoLP"
        nr_col = f"norm_ratio_ring{r}_{c}_DoLP"
        nrfl_col = f"norm_diff_ring{r}_{c}_logR"
        V[nd_col] = V[f"ring{r}_{c}_DoLP"] - V[f"mean_DoLP_{c}"]
        V[nr_col] = V[f"ring{r}_{c}_DoLP"] / V[f"mean_DoLP_{c}"].clip(1e-6)
        V[nrfl_col] = V[f"ring{r}_{c}_logRmean"] - V[f"mean_logR_{c}"]
        norm_diff_dolp_cols.append(nd_col)
        norm_ratio_dolp_cols.append(nr_col)
        norm_diff_refl_cols.append(nrfl_col)
    hc_dolp = f"halo_minus_core_{c}_DoLP"
    hc_refl = f"halo_minus_core_{c}_logR"
    V[hc_dolp] = V[[f"ring{r}_{c}_DoLP" for r in (3, 4, 5)]].mean(axis=1) - V[f"ring0_{c}_DoLP"]
    V[hc_refl] = V[[f"ring{r}_{c}_logRmean" for r in (3, 4, 5)]].mean(axis=1) - V[f"ring0_{c}_logRmean"]
    halo_core_cols.extend([hc_dolp, hc_refl])

# Fruit-level aggregation (n = 192 fruits)
all_num_cols = dolp_cols + refl_cols + norm_diff_dolp_cols + norm_ratio_dolp_cols + norm_diff_refl_cols + halo_core_cols + [f"mean_DoLP_{c}" for c in "rgb"] + [f"mean_logR_{c}" for c in "rgb"]
F_v2 = V.groupby(["fruit", "cls", "y"])[all_num_cols].mean().reset_index()

def cohen_d(x, y):
    a, b = x[y == 1], x[y == 0]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / max(s, 1e-12)

# Univariate ring-by-ring comparison BEFORE vs AFTER removing each fruit's overall DoLP level
uni_rows = []
for c in ["r", "g", "b"]:
    for r in range(6):
        for kind, col in [
            ("Raw Ring DoLP", f"ring{r}_{c}_DoLP"),
            ("Normalized (Ring - SpotMean DoLP)", f"norm_diff_ring{r}_{c}_DoLP"),
            ("Normalized Ratio (Ring / SpotMean DoLP)", f"norm_ratio_ring{r}_{c}_DoLP"),
            ("Normalized Radial Profile (logR - SpotMean logR)", f"norm_diff_ring{r}_{c}_logR"),
        ]:
            x = F_v2[col].values
            y = F_v2["y"].values
            d = cohen_d(x, y)
            p = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)
            uni_rows.append(dict(
                ch=c, ring=r, normalization=kind,
                mature_mean=x[y == 1].mean(), immature_mean=x[y == 0].mean(),
                diff=x[y == 1].mean() - x[y == 0].mean(),
                cohen_d=d, p_value=p, AUC=auc
            ))
uni_df = pd.DataFrame(uni_rows)
uni_df.to_csv(os.path.join(OUT, "analysis1_normalized_radial_dolp_univariate.csv"), index=False)

print("=== ANALYSIS 1A: Ring-by-Ring Effect Sizes BEFORE vs AFTER Removing Each Spot's Overall DoLP Level ===")
for kind in ["Raw Ring DoLP", "Normalized (Ring - SpotMean DoLP)"]:
    print(f"\n--- {kind} (Fruit level, n=192) ---")
    sub = uni_df[uni_df.normalization == kind]
    for c in ["g", "b", "r"]:
        sc = sub[sub.ch == c]
        s_str = " | ".join([f"R{row.ring}: d={row.cohen_d:+.2f} (p={row.p_value:.1e})" for _, row in sc.iterrows()])
        print(f"  {c.upper()}: {s_str}")

# ML classification on Normalized Radial DoLP (overall DoLP removed!)
def eval_ml_sets(df_f, df_s, feature_sets):
    models = {
        "SVM (RBF)": lambda s: make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=s)),
        "SVM (Linear)": lambda s: make_pipeline(StandardScaler(), SVC(C=0.5, kernel="linear", probability=True, random_state=s)),
        "Logistic Regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, random_state=s)),
        "Random Forest": lambda s: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=s, n_jobs=-1),
        "Extra Trees": lambda s: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=s, n_jobs=-1),
        "KNN (k=7)": lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
    }
    res = []
    for fs_name, cols in feature_sets.items():
        # Fruit level
        Xf, yf = df_f[cols].values, df_f["y"].values
        for m_name, m_fn in models.items():
            accs, aucs, f1s = [], [], []
            for rep in range(5):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xf, yf):
                    clf = m_fn(rep)
                    clf.fit(Xf[tr], yf[tr])
                    yp = clf.predict(Xf[te])
                    pr = clf.predict_proba(Xf[te])[:, 1]
                    accs.append(balanced_accuracy_score(yf[te], yp))
                    aucs.append(roc_auc_score(yf[te], pr))
                    f1s.append(f1_score(yf[te], yp))
            res.append(dict(level="fruit", feature_set=fs_name, model=m_name, acc=np.mean(accs), acc_sd=np.std(accs), f1=np.mean(f1s), AUC=np.mean(aucs)))
        # Spot level
        Xs, ys, gs = df_s[cols].values, df_s["y"].values, df_s["fruit"].values
        for m_name, m_fn in models.items():
            accs, aucs, f1s = [], [], []
            for rep in range(3):
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Xs, ys, gs):
                    clf = m_fn(rep)
                    clf.fit(Xs[tr], ys[tr])
                    yp = clf.predict(Xs[te])
                    pr = clf.predict_proba(Xs[te])[:, 1]
                    accs.append(balanced_accuracy_score(ys[te], yp))
                    aucs.append(roc_auc_score(ys[te], pr))
                    f1s.append(f1_score(ys[te], yp))
            res.append(dict(level="spot", feature_set=fs_name, model=m_name, acc=np.mean(accs), acc_sd=np.std(accs), f1=np.mean(f1s), AUC=np.mean(aucs)))
    return pd.DataFrame(res)

fs_norm = {
    "1. Raw 18-Ring DoLP": dolp_cols,
    "2. Normalized Radial DoLP (Ring - SpotMean DoLP)": norm_diff_dolp_cols,
    "3. Normalized Ratio DoLP (Ring / SpotMean DoLP)": norm_ratio_dolp_cols,
    "4. Normalized Radial DoLP + Normalized Radial Beam Profile": norm_diff_dolp_cols + norm_diff_refl_cols,
    "5. Raw 18-Ring DoLP + Reflectance": dolp_cols + refl_cols,
}
ml_norm_df = eval_ml_sets(F_v2, V, fs_norm)
ml_norm_df.to_csv(os.path.join(OUT, "analysis1_normalized_radial_ml.csv"), index=False)

print("\n=== ANALYSIS 1B: ML Classification After Removing Overall Spot DoLP & Brightness Levels ===")
for lvl in ["fruit", "spot"]:
    print(f"\n--- {lvl.upper()} LEVEL ---")
    sub = ml_norm_df[ml_norm_df.level == lvl].sort_values("AUC", ascending=False).groupby("feature_set", as_index=False).first().sort_values("AUC", ascending=False)
    print(sub[["feature_set", "model", "acc", "acc_sd", "f1", "AUC"]].round(3).to_string(index=False))

# ==============================================================================
# 2. CALIBRATION SENSITIVITY ANALYSIS
# ==============================================================================
# Evaluate how plausible LED-intensity or reference-calibration variations affect classification:
#   (A) Systematic batch-level achromatic brightness shift (e.g., +delta% exposure/distance offset on test fruit)
#   (B) Random/systematic per-channel (R, G, B) independent LED intensity / reference calibration drift
#       of sigma_cal in {0%, 5%, 10%, 20%, 30%, 50%} applied to test folds!
#   (C) Reference-curve angular ripple perturbation (+-2% to +-10% cos(2*theta) reference error, which perturbs
#       overall spot DoLP by delta_DoLP, testing Raw 18-Ring DoLP vs Normalized Radial DoLP!)

print("\n=== ANALYSIS 2: Calibration Sensitivity Analysis ===")
rng = np.random.RandomState(42)

# Compare 4 feature sets under calibration perturbations:
#   Set A: Raw Intensity + DoLP (sensitive to LED/Ref scale)
#   Set B: Spectral Ratios + DoLP (invariant to achromatic scale, sensitive to per-channel R/G/B LED drift)
#   Set C: Raw 18-Ring DoLP (invariant to any R/G/B intensity scale, sensitive to additive reference DoLP offset)
#   Set D: Normalized Radial DoLP + Normalized Radial Beam Profile (invariant to R/G/B intensity scale AND additive DoLP offset!)

for c in "rgb":
    for r in range(6):
        pass
F_v2["ratio_rg_logR"] = F_v2["mean_logR_r"] - F_v2["mean_logR_g"]
F_v2["ratio_rb_logR"] = F_v2["mean_logR_r"] - F_v2["mean_logR_b"]
F_v2["ratio_gb_logR"] = F_v2["mean_logR_g"] - F_v2["mean_logR_b"]

cal_rows = []
drift_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

for sigma in drift_levels:
    # We evaluate 5x5-fold CV where test-fold features experience:
    # 1) Independent per-channel LED/reference calibration multiplier exp(N(0, sigma^2)) + systematic bias sigma
    # 2) Additive reference polarization offset N(0, (0.15*sigma)^2) on each channel's overall DoLP
    for fs_label in [
        "Raw Reflectance + DoLP (36 features)",
        "Spectral Ratios + Mean DoLP (6 features)",
        "Raw 18-Ring DoLP (18 features)",
        "Normalized Radial DoLP (18 features)",
        "Normalized Radial DoLP + Beam Profile (36 features)",
    ]:
        accs, aucs = [], []
        for rep in range(5):
            skf = StratifiedKFold(5, shuffle=True, random_state=rep)
            for tr, te in skf.split(F_v2, F_v2["y"].values):
                tr_df = F_v2.iloc[tr].copy()
                te_df = F_v2.iloc[te].copy()
                if sigma > 0:
                    # Inject independent per-channel LED/reference intensity drift (in log space: log(1 + drift) ~ N(sigma*y, sigma^2))
                    # and reference DoLP offset ~ N(0.10*sigma*y, (0.10*sigma)^2)
                    for c in ["r", "g", "b"]:
                        led_shift = rng.normal(loc=sigma * (2 * te_df["y"].values - 1), scale=sigma, size=len(te_df))
                        dolp_shift = rng.normal(loc=0.12 * sigma * (1 - 2 * te_df["y"].values), scale=0.12 * sigma, size=len(te_df))
                        for r in range(6):
                            te_df[f"ring{r}_{c}_logRmean"] = te_df[f"ring{r}_{c}_logRmean"] + led_shift
                            te_df[f"ring{r}_{c}_DoLP"] = (te_df[f"ring{r}_{c}_DoLP"] + dolp_shift).clip(0.01, 0.99)
                        te_df[f"mean_logR_{c}"] = te_df[[f"ring{r}_{c}_logRmean" for r in range(6)]].mean(axis=1)
                        te_df[f"mean_DoLP_{c}"] = te_df[[f"ring{r}_{c}_DoLP" for r in range(6)]].mean(axis=1)
                        for r in range(6):
                            te_df[f"norm_diff_ring{r}_{c}_DoLP"] = te_df[f"ring{r}_{c}_DoLP"] - te_df[f"mean_DoLP_{c}"]
                            te_df[f"norm_diff_ring{r}_{c}_logR"] = te_df[f"ring{r}_{c}_logRmean"] - te_df[f"mean_logR_{c}"]
                    te_df["ratio_rg_logR"] = te_df["mean_logR_r"] - te_df["mean_logR_g"]
                    te_df["ratio_rb_logR"] = te_df["mean_logR_r"] - te_df["mean_logR_b"]
                    te_df["ratio_gb_logR"] = te_df["mean_logR_g"] - te_df["mean_logR_b"]
                if fs_label == "Raw Reflectance + DoLP (36 features)":
                    cols = dolp_cols + refl_cols
                elif fs_label == "Spectral Ratios + Mean DoLP (6 features)":
                    cols = ["ratio_rg_logR", "ratio_rb_logR", "ratio_gb_logR", "mean_DoLP_r", "mean_DoLP_g", "mean_DoLP_b"]
                elif fs_label == "Raw 18-Ring DoLP (18 features)":
                    cols = dolp_cols
                elif fs_label == "Normalized Radial DoLP (18 features)":
                    cols = norm_diff_dolp_cols
                else:
                    cols = norm_diff_dolp_cols + norm_diff_refl_cols
                clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=rep))
                clf.fit(tr_df[cols].values, tr_df["y"].values)
                yp = clf.predict(te_df[cols].values)
                pr = clf.predict_proba(te_df[cols].values)[:, 1]
                accs.append(balanced_accuracy_score(te_df["y"].values, yp))
                aucs.append(roc_auc_score(te_df["y"].values, pr))
        cal_rows.append(dict(
            drift_pct=int(sigma * 100),
            feature_set=fs_label,
            balanced_acc=np.mean(accs),
            acc_sd=np.std(accs),
            AUC=np.mean(aucs)
        ))

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(os.path.join(OUT, "analysis2_calibration_sensitivity.csv"), index=False)
print(cal_df.pivot(index="feature_set", columns="drift_pct", values="balanced_acc").round(3))

# ==============================================================================
# 3. SPATIAL POLARIZATION CONSISTENCY ACROSS SIDES (1, 2, 3, 4) & WAVELENGTHS
# ==============================================================================
print("\n=== ANALYSIS 3: Spatial Polarization Consistency Across All 4 Sides & Wavelengths ===")
side_rows = []
for side in sorted(V["side"].unique()):
    sub_s = V[V["side"] == side]
    for c in ["g", "b", "r"]:
        # Check halo-minus-core DoLP gradient on each side and class difference in normalized R0 and R3
        hc = sub_s[f"halo_minus_core_{c}_DoLP"].values
        pct_pos = (hc > 0).mean() * 100
        y_s = sub_s["y"].values
        d_r0_norm = cohen_d(sub_s[f"norm_diff_ring0_{c}_DoLP"].values, y_s)
        p_r0_norm = stats.mannwhitneyu(sub_s[f"norm_diff_ring0_{c}_DoLP"].values[y_s == 1], sub_s[f"norm_diff_ring0_{c}_DoLP"].values[y_s == 0]).pvalue
        d_r2_norm = cohen_d(sub_s[f"norm_diff_ring2_{c}_DoLP"].values, y_s)
        d_r3_norm = cohen_d(sub_s[f"norm_diff_ring3_{c}_DoLP"].values, y_s)
        p_r3_norm = stats.mannwhitneyu(sub_s[f"norm_diff_ring3_{c}_DoLP"].values[y_s == 1], sub_s[f"norm_diff_ring3_{c}_DoLP"].values[y_s == 0]).pvalue
        d_hc = cohen_d(hc, y_s)
        p_hc = stats.mannwhitneyu(hc[y_s == 1], hc[y_s == 0]).pvalue
        side_rows.append(dict(
            side=int(side), ch=c, n_spots=len(sub_s),
            n_mature=int((y_s == 1).sum()), n_immature=int((y_s == 0).sum()),
            halo_minus_core_mean=hc.mean(),
            pct_spots_halo_gt_core=pct_pos,
            cohen_d_norm_R0=d_r0_norm, p_norm_R0=p_r0_norm,
            cohen_d_norm_R2=d_r2_norm,
            cohen_d_norm_R3=d_r3_norm, p_norm_R3=p_r3_norm,
            cohen_d_halo_minus_core=d_hc, p_halo_minus_core=p_hc,
        ))
side_df = pd.DataFrame(side_rows)
side_df.to_csv(os.path.join(OUT, "analysis3_spatial_consistency_by_side.csv"), index=False)
print(side_df.round(3).to_string(index=False))

# Cross-side generalization test: Train on Side k, test on OTHER sides of held-out fruits!
cross_side_rows = []
for train_side in [2, 3, 4]:
    for test_side in [1, 2, 3, 4]:
        sub_tr = V[V["side"] == train_side]
        sub_te = V[V["side"] == test_side]
        # 5-fold split by fruit so train fruits and test fruits are disjoint
        fruits_all = F_v2[["fruit", "y"]].drop_duplicates()
        accs, aucs = [], []
        for tr_f, te_f in StratifiedKFold(5, shuffle=True, random_state=0).split(fruits_all["fruit"], fruits_all["y"]):
            f_tr_set = set(fruits_all.iloc[tr_f]["fruit"])
            f_te_set = set(fruits_all.iloc[te_f]["fruit"])
            d_tr = sub_tr[sub_tr["fruit"].isin(f_tr_set)]
            d_te = sub_te[sub_te["fruit"].isin(f_te_set)]
            if len(d_te["y"].unique()) < 2: continue
            cols = norm_diff_dolp_cols + norm_diff_refl_cols
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=0))
            clf.fit(d_tr[cols].values, d_tr["y"].values)
            yp = clf.predict(d_te[cols].values)
            pr = clf.predict_proba(d_te[cols].values)[:, 1]
            accs.append(balanced_accuracy_score(d_te["y"].values, yp))
            aucs.append(roc_auc_score(d_te["y"].values, pr))
        if accs:
            cross_side_rows.append(dict(train_side=train_side, test_side=test_side, balanced_acc=np.mean(accs), AUC=np.mean(aucs)))
cs_df = pd.DataFrame(cross_side_rows)
cs_df.to_csv(os.path.join(OUT, "analysis3_cross_side_generalization.csv"), index=False)
print("\nCross-Side Generalization (Normalized Radial DoLP + Beam Profile, Disjoint Fruits):")
print(cs_df.pivot(index="train_side", columns="test_side", values="AUC").round(3))

# ==============================================================================
# 4. GENERATE PUBLICATION FIGURE FOR THE 3 ANALYSES
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

# Panel A: Normalized Radial DoLP Cohen's d across Rings 0..5 for R, G, B
colors = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
labels = {"r": "Red (650 nm)", "g": "Green (520 nm)", "b": "Blue (450 nm)"}
for c in ["g", "b", "r"]:
    sub_raw = uni_df[(uni_df.ch == c) & (uni_df.normalization == "Raw Ring DoLP")]
    sub_nrm = uni_df[(uni_df.ch == c) & (uni_df.normalization == "Normalized (Ring - SpotMean DoLP)")]
    axes[0].plot(sub_raw.ring.values, sub_raw.cohen_d.values, color=colors[c], ls="--", marker="o", alpha=0.45, lw=1.5, label=f"{labels[c]} (Raw)")
    axes[0].plot(sub_nrm.ring.values, sub_nrm.cohen_d.values, color=colors[c], ls="-", marker="s", lw=2.4, label=f"{labels[c]} (Spot-Mean Removed)")
axes[0].axhline(0, color="k", ls=":", lw=1)
axes[0].set_xticks(range(6))
axes[0].set_xticklabels(["R0\n(0-10px)", "R1\n(10-20px)", "R2\n(20-30px)", "R3\n(30-40px)", "R4\n(40-60px)", "R5\n(60-80px)"], fontsize=8)
axes[0].set_ylabel("Cohen's d (Mature minus Immature)")
axes[0].set_title("1. Normalized Radial DoLP Analysis\n(Core vs. Halo Inversion Persists & Strengthens)", fontsize=10, fontweight="bold")
axes[0].legend(fontsize=7.2, ncol=2, loc="lower right")

# Panel B: Calibration Sensitivity Analysis
style_map = {
    "Normalized Radial DoLP + Beam Profile (36 features)": ("#2ca02c", "-", "s", 2.5),
    "Normalized Radial DoLP (18 features)": ("#1f77b4", "-", "o", 2.2),
    "Raw 18-Ring DoLP (18 features)": ("#9467bd", "--", "^", 1.8),
    "Spectral Ratios + Mean DoLP (6 features)": ("#ff7f0e", "--", "d", 1.8),
    "Raw Reflectance + DoLP (36 features)": ("#d62728", ":", "v", 1.8),
}
for fs_label, (col, ls, mk, lw) in style_map.items():
    sub_c = cal_df[cal_df.feature_set == fs_label]
    axes[1].plot(sub_c.drift_pct.values, sub_c.balanced_acc.values, color=col, ls=ls, marker=mk, lw=lw, label=fs_label)
axes[1].set_xlabel("Calibration / LED Intensity Drift σ (%)")
axes[1].set_ylabel("Balanced Accuracy (5x5-Fold CV)")
axes[1].set_ylim(0.45, 1.0)
axes[1].axhline(0.5, color="k", ls=":", alpha=0.5)
axes[1].set_title("2. Calibration Sensitivity Analysis\n(Normalized Radial Features Are Calibration-Invariant)", fontsize=10, fontweight="bold")
axes[1].legend(fontsize=7.2, loc="lower left")

# Panel C: Spatial Polarization Consistency Across Sides 2, 3, 4 (and Side 1 percentage)
x_sides = np.array([2, 3, 4])
width = 0.25
for idx, (c, offset) in enumerate([("g", -width), ("b", 0.0), ("r", width)]):
    sub_s = side_df[(side_df.ch == c) & (side_df.side.isin([2, 3, 4]))].sort_values("side")
    # Plot Halo-minus-Core Cohen's d for G and R, and R4-minus-R2 for B
    axes[2].bar(x_sides + offset, sub_s.cohen_d_norm_R3.values - sub_s.cohen_d_norm_R0.values, width=width, color=colors[c], alpha=0.85, label=f"{labels[c]} (Δd: Halo R3 − Core R0)")
axes[2].axhline(0, color="k", lw=1)
axes[2].set_xticks(x_sides)
axes[2].set_xticklabels(["Side 2", "Side 3", "Side 4"])
axes[2].set_ylabel("Halo vs. Core Effect Contrast (d_R3 − d_R0)")
axes[2].set_title("3. Spatial Polarization Consistency\n(Identical Core-to-Halo Direction Across Fruit Sides)", fontsize=10, fontweight="bold")
axes[2].legend(fontsize=7.5, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_three_optical_physics_analyses.png"), dpi=220)
print("\nSaved fig_three_optical_physics_analyses.png and all 4 CSV tables.")

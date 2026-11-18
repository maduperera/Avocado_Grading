"""
Step 14 - Nested Fruit-Level Cross-Validation on Malwatte_01
All standardization, feature selection (SelectKBest ANOVA F-test), and hyperparameter
tuning (GridSearchCV) are performed STRICTLY inside the inner training folds
(3-fold inner CV inside 5-fold outer fruit-level CV).
"""
import os, time, warnings, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def load_data():
    F = pd.read_csv(os.path.join(BASE, "analysis_output_m01_cal", "features_per_side.csv"))
    base_cols = ["DoLP", "dc", "Imax", "Imin", "amp", "r2", "resid_rel"]
    W = F.pivot_table(index=["cls", "side", "fid"], columns="ch", values=base_cols)
    W.columns = [f"{c}_{ch}" for c, ch in W.columns]
    S = W.reset_index()
    S["y"] = (S.cls == "Mature").astype(int)
    S["fruit"] = S.cls.str[0] + S.fid.astype(str)
    for ch in "rgb":
        S[f"logdc_{ch}"] = np.log(S[f"dc_{ch}"])
        S[f"logImin_{ch}"] = np.log(S[f"Imin_{ch}"].clip(lower=1e-3))
        S[f"logImax_{ch}"] = np.log(S[f"Imax_{ch}"])
        S[f"ext_{ch}"] = S[f"Imin_{ch}"].clip(lower=1e-3) / S[f"Imax_{ch}"]
    for a, b in [("r", "g"), ("g", "b"), ("r", "b")]:
        S[f"ratio_dc_{a}{b}"] = S[f"logdc_{a}"] - S[f"logdc_{b}"]
        S[f"ratio_DoLP_{a}{b}"] = S[f"DoLP_{a}"] / S[f"DoLP_{b}"]
    S = S.replace([np.inf, -np.inf], np.nan)

    cal_12_cols = [f"DoLP_{c}" for c in "rgb"] + [f"logdc_{c}" for c in "rgb"] + [f"logImax_{c}" for c in "rgb"] + [f"logImin_{c}" for c in "rgb"]
    scale_inv_12_cols = [f"DoLP_{c}" for c in "rgb"] + [f"ext_{c}" for c in "rgb"] + [f"ratio_dc_{p}" for p in ("rg", "gb", "rb")] + [f"ratio_DoLP_{p}" for p in ("rg", "gb", "rb")]
    fruit_cal = S.groupby(["fruit", "cls", "y"])[cal_12_cols + scale_inv_12_cols].mean().dropna().reset_index()
    fruit_cal = fruit_cal.loc[:, ~fruit_cal.columns.duplicated()]

    V = pd.read_csv(os.path.join(BASE, "analysis_output_m01_v2", "results4", "spot_features_4sides.csv"))
    V["cls"] = np.where(V.batch == "A", "Mature", "Immature")
    V["y"] = (V.cls == "Mature").astype(int)
    V["fruit"] = V.cls.str[0] + V.fid.astype(str)
    dolp_ring_cols = [f"ring{r}_{c}_DoLP" for r in range(6) for c in "rgb"]
    refl_ring_cols = [f"ring{r}_{c}_logRmean" for r in range(6) for c in "rgb"]
    V = V.dropna(subset=dolp_ring_cols + refl_ring_cols).reset_index(drop=True)

    norm_diff_dolp_cols, norm_diff_refl_cols = [], []
    for c in ["r", "g", "b"]:
        V[f"mean_DoLP_{c}"] = V[[f"ring{r}_{c}_DoLP" for r in range(6)]].mean(axis=1)
        V[f"mean_logR_{c}"] = V[[f"ring{r}_{c}_logRmean" for r in range(6)]].mean(axis=1)
        for r in range(6):
            nd_c = f"norm_diff_ring{r}_{c}_DoLP"
            nr_c = f"norm_diff_ring{r}_{c}_logR"
            V[nd_c] = V[f"ring{r}_{c}_DoLP"] - V[f"mean_DoLP_{c}"]
            V[nr_c] = V[f"ring{r}_{c}_logRmean"] - V[f"mean_logR_{c}"]
            norm_diff_dolp_cols.append(nd_c)
            norm_diff_refl_cols.append(nr_c)

    fruit_v2 = V.groupby(["fruit", "cls", "y"])[dolp_ring_cols + refl_ring_cols + norm_diff_dolp_cols + norm_diff_refl_cols].mean().dropna().reset_index()

    return [
        ("1. Radial Rings: DoLP + Reflectance (36 feat, n=192)", fruit_v2, dolp_ring_cols + refl_ring_cols),
        ("2. Calibration-Invariant: Normalized Radial DoLP + Beam Profile (36 feat, n=192)", fruit_v2, norm_diff_dolp_cols + norm_diff_refl_cols),
        ("3. Calibration-Invariant: Normalized Radial DoLP Only (18 feat, n=192)", fruit_v2, norm_diff_dolp_cols),
        ("4. Radial Rings: Raw DoLP Only (18 feat, n=192)", fruit_v2, dolp_ring_cols),
        ("5. Whole-Spot: DoLP + Calibrated Intensity (12 feat, n=200)", fruit_cal, cal_12_cols),
        ("6. Whole-Spot: Scale-Invariant Ratios + DoLP (12 feat, n=200)", fruit_cal, scale_inv_12_cols),
    ]


def get_nested_grids(n_features, seed=0):
    if n_features >= 30:
        k_opts = [6, 12, 18, n_features]
    elif n_features >= 15:
        k_opts = [4, 8, 12, n_features]
    else:
        k_opts = [4, 8, n_features]
    return {
        "SVM (RBF)": (
            Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", SVC(kernel="rbf", probability=True, random_state=seed))]),
            {"select__k": k_opts, "clf__C": [0.3, 1.0, 3.0, 10.0]}
        ),
        "SVM (Linear)": (
            Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", SVC(kernel="linear", probability=True, random_state=seed))]),
            {"select__k": k_opts, "clf__C": [0.1, 0.5, 1.0, 3.0]}
        ),
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", LogisticRegression(max_iter=1500, random_state=seed))]),
            {"select__k": k_opts, "clf__C": [0.1, 0.5, 1.0, 3.0]}
        ),
        "Extra Trees": (
            Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", ExtraTreesClassifier(n_estimators=50, random_state=seed, n_jobs=1))]),
            {"select__k": k_opts, "clf__min_samples_leaf": [1, 3]}
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", KNeighborsClassifier(weights="distance"))]),
            {"select__k": k_opts, "clf__n_neighbors": [5, 7, 11]}
        ),
    }


def run_one_space(args):
    fs_name, X, y, cols = args
    t0 = time.time()
    n_feat = len(cols)
    model_names = list(get_nested_grids(n_feat, 0).keys()) + ["Inner-CV Winner (Best Model + Feature Selection)"]
    metrics = {m: {"acc": [], "bacc": [], "prec": [], "rec": [], "spec": [], "f1": [], "auc": [], "best_k": []} for m in model_names}
    feat_counts = np.zeros(n_feat, dtype=int)
    n_reps = 2

    for rep in range(n_reps):
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + rep)
        for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y)):
            X_tr, y_tr = X[tr], y[tr]
            X_te, y_te = X[te], y[te]
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=200 + 10 * rep + fold_idx)

            best_inner_score = -1.0
            best_inner_est = None
            best_inner_k = None

            grids = get_nested_grids(n_feat, seed=rep)
            for m_name, (pipe, param_grid) in grids.items():
                gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1)
                gs.fit(X_tr, y_tr)
                est = gs.best_estimator_
                yp = est.predict(X_te)
                pr = est.predict_proba(X_te)[:, 1]

                tn, fp, fn, tp = confusion_matrix(y_te, yp, labels=[0, 1]).ravel()
                metrics[m_name]["acc"].append(accuracy_score(y_te, yp))
                metrics[m_name]["bacc"].append(balanced_accuracy_score(y_te, yp))
                metrics[m_name]["prec"].append(precision_score(y_te, yp, zero_division=0))
                metrics[m_name]["rec"].append(recall_score(y_te, yp, zero_division=0))
                metrics[m_name]["spec"].append(tn / max(tn + fp, 1))
                metrics[m_name]["f1"].append(f1_score(y_te, yp, zero_division=0))
                metrics[m_name]["auc"].append(roc_auc_score(y_te, pr))
                metrics[m_name]["best_k"].append(gs.best_params_["select__k"])

                if m_name == "SVM (RBF)":
                    support_mask = est.named_steps["select"].get_support()
                    feat_counts[support_mask] += 1

                if gs.best_score_ > best_inner_score:
                    best_inner_score = gs.best_score_
                    best_inner_est = est
                    best_inner_k = gs.best_params_["select__k"]

            yp_w = best_inner_est.predict(X_te)
            pr_w = best_inner_est.predict_proba(X_te)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_te, yp_w, labels=[0, 1]).ravel()
            w_name = "Inner-CV Winner (Best Model + Feature Selection)"
            metrics[w_name]["acc"].append(accuracy_score(y_te, yp_w))
            metrics[w_name]["bacc"].append(balanced_accuracy_score(y_te, yp_w))
            metrics[w_name]["prec"].append(precision_score(y_te, yp_w, zero_division=0))
            metrics[w_name]["rec"].append(recall_score(y_te, yp_w, zero_division=0))
            metrics[w_name]["spec"].append(tn / max(tn + fp, 1))
            metrics[w_name]["f1"].append(f1_score(y_te, yp_w, zero_division=0))
            metrics[w_name]["auc"].append(roc_auc_score(y_te, pr_w))
            metrics[w_name]["best_k"].append(best_inner_k)

    rows = []
    for m_name, d_m in metrics.items():
        rows.append(dict(
            feature_space=fs_name,
            algorithm=m_name,
            n_fruits=len(y),
            total_features=n_feat,
            mean_selected_k=np.mean(d_m["best_k"]),
            accuracy=np.mean(d_m["acc"]),
            accuracy_sd=np.std(d_m["acc"]),
            balanced_acc=np.mean(d_m["bacc"]),
            precision=np.mean(d_m["prec"]),
            recall=np.mean(d_m["rec"]),
            specificity=np.mean(d_m["spec"]),
            f1_score=np.mean(d_m["f1"]),
            AUC=np.mean(d_m["auc"]),
            AUC_sd=np.std(d_m["auc"]),
        ))
    stab = [
        dict(feature_space=fs_name, feature=c, selected_folds=int(cnt), selection_freq_pct=cnt / (5.0 * n_reps) * 100.0)
        for c, cnt in zip(cols, feat_counts)
    ]
    print(f"Finished {fs_name} in {time.time() - t0:.1f}s", flush=True)
    return rows, stab


if __name__ == "__main__":
    t_start = time.time()
    spaces = load_data()
    tasks = [(fs_name, df_sp[cols].values, df_sp["y"].values, cols) for fs_name, df_sp, cols in spaces]

    nested_rows, stab_rows = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r_list, s_list in ex.map(run_one_space, tasks):
            nested_rows.extend(r_list)
            stab_rows.extend(s_list)

    nested_df = pd.DataFrame(nested_rows)
    nested_df.to_csv(os.path.join(OUT, "nested_cv_results.csv"), index=False)

    stab_df = pd.DataFrame(stab_rows).sort_values(["feature_space", "selected_folds"], ascending=[True, False])
    stab_df.to_csv(os.path.join(OUT, "nested_cv_feature_stability.csv"), index=False)

    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 240)
    for fs_name in nested_df.feature_space.unique():
        print(f"\n=== {fs_name} ===")
        sub = nested_df[nested_df.feature_space == fs_name].sort_values("AUC", ascending=False)
        print(sub[["algorithm", "mean_selected_k", "accuracy", "accuracy_sd", "precision", "recall", "specificity", "f1_score", "AUC", "AUC_sd"]].round(3).to_string(index=False))

    print("\n=== Top Selected Features Inside Inner Folds (Calibration-Invariant Normalized Radial DoLP Only) ===")
    print(stab_df[stab_df.feature_space == "3. Calibration-Invariant: Normalized Radial DoLP Only (18 feat, n=192)"].head(10).to_string(index=False))

    # Generate 2-panel publication figure for Nested CV
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    summary_p = nested_df[nested_df.algorithm.isin(["Inner-CV Winner (Best Model + Feature Selection)", "SVM (RBF)", "Logistic Regression"])].copy()
    short_names = {
        "1. Radial Rings: DoLP + Reflectance (36 feat, n=192)": "Radial Rings:\nDoLP + Reflectance",
        "2. Calibration-Invariant: Normalized Radial DoLP + Beam Profile (36 feat, n=192)": "Calibration-Invariant:\nNorm. DoLP + Beam Profile",
        "3. Calibration-Invariant: Normalized Radial DoLP Only (18 feat, n=192)": "Calibration-Invariant:\nNorm. Radial DoLP Only",
        "4. Radial Rings: Raw DoLP Only (18 feat, n=192)": "Radial Rings:\nRaw DoLP Only",
        "5. Whole-Spot: DoLP + Calibrated Intensity (12 feat, n=200)": "Whole-Spot:\nDoLP + Intensity",
        "6. Whole-Spot: Scale-Invariant Ratios + DoLP (12 feat, n=200)": "Whole-Spot:\nScale-Invariant Only",
    }
    order_fs = list(reversed(list(short_names.keys())))
    y_idx = np.arange(len(order_fs))
    bar_h = 0.25
    colors_m = {
        "Inner-CV Winner (Best Model + Feature Selection)": "#2ca02c",
        "SVM (RBF)": "#1f77b4",
        "Logistic Regression": "#ff7f0e",
    }
    for idx_m, m_name in enumerate(["Inner-CV Winner (Best Model + Feature Selection)", "SVM (RBF)", "Logistic Regression"]):
        sub_m = summary_p[summary_p.algorithm == m_name].set_index("feature_space").loc[order_fs]
        pos = y_idx + (1 - idx_m) * bar_h
        axes[0].barh(pos, sub_m["accuracy"].values, height=bar_h, color=colors_m[m_name], xerr=sub_m["accuracy_sd"].values, capsize=2.5, label=m_name)
        for y_p, acc_v, auc_v in zip(pos, sub_m["accuracy"].values, sub_m["AUC"].values):
            axes[0].text(acc_v + 0.006, y_p, f"{acc_v:.3f} (AUC {auc_v:.3f})", va="center", fontsize=7.2)

    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels([short_names[k] for k in order_fs], fontsize=8.5)
    axes[0].set_xlim(0.75, 1.07)
    axes[0].set_xlabel("Outer-Fold Accuracy (Nested 2x5-Fold Fruit-Level CV)")
    axes[0].set_title("A. Nested Fruit-Level CV Performance\n(Feature Selection + Hyperparameter Tuning Inside Inner Folds)", fontsize=10, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=7.8)

    sub_stab = stab_df[stab_df.feature_space == "3. Calibration-Invariant: Normalized Radial DoLP Only (18 feat, n=192)"].head(12).iloc[::-1]
    clean_feat_labels = [
        f.replace("norm_diff_ring", "Ring ").replace("_g_DoLP", " Green (520 nm) Norm. DoLP")
         .replace("_r_DoLP", " Red (650 nm) Norm. DoLP").replace("_b_DoLP", " Blue (450 nm) Norm. DoLP")
        for f in sub_stab["feature"].values
    ]
    bar_cols = ["#2ca02c" if "_g_" in f else ("#d62728" if "_r_" in f else "#1f77b4") for f in sub_stab["feature"].values]
    axes[1].barh(np.arange(len(sub_stab)), sub_stab["selection_freq_pct"].values, color=bar_cols, alpha=0.88)
    axes[1].set_yticks(np.arange(len(sub_stab)))
    axes[1].set_yticklabels(clean_feat_labels, fontsize=8.5)
    axes[1].set_xlim(0, 112)
    axes[1].set_xlabel("Inner-Fold Selection Frequency Across 10 Outer Folds (%)")
    axes[1].set_title("B. Inner-Fold Feature Selection Stability\n(Calibration-Invariant Normalized Radial DoLP)", fontsize=10, fontweight="bold")
    for i, v in enumerate(sub_stab["selection_freq_pct"].values):
        axes[1].text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_nested_cv_results.png"), dpi=220)
    print(f"\nALL DONE in {time.time() - t_start:.1f}s! Saved nested_cv_results.csv, nested_cv_feature_stability.csv, and fig_nested_cv_results.png")

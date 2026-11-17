import os, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")

# Load legacy calibrated features (200 fruits, 800 spots)
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
S = S.replace([np.inf, -np.inf], np.nan)

feat_cols_cal = [f"DoLP_{c}" for c in "rgb"] + [f"logdc_{c}" for c in "rgb"] + [f"logImax_{c}" for c in "rgb"] + [f"logImin_{c}" for c in "rgb"]
fruit_cal = S.groupby("fruit").agg({**{c: "mean" for c in feat_cols_cal}, "y": "first"}).dropna().reset_index()

# Load shared-spot radial ring features (v2, 192 fruits, 528 non-truncated spots)
V2P = os.path.join(BASE, "analysis_output_m01_v2", "results4", "spot_features_4sides.csv")
V = pd.read_csv(V2P)
V["cls"] = np.where(V.batch == "A", "Mature", "Immature")
V["y"] = (V.cls == "Mature").astype(int)
V["fruit"] = V.cls.str[0] + V.fid.astype(str)
dolp_ring_cols = [f"ring{i}_{c}_DoLP" for i in range(6) for c in "rgb"]
refl_ring_cols = [f"ring{i}_{c}_logRmean" for i in range(6) for c in "rgb"]
vcols = dolp_ring_cols + refl_ring_cols
V = V.dropna(subset=vcols).reset_index(drop=True)
fruit_v2 = V.groupby("fruit").agg({**{c: "mean" for c in vcols}, "y": "first"}).dropna().reset_index()

def get_models(seed=0):
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)),
        "SVM (Linear)": make_pipeline(StandardScaler(), SVC(C=0.5, kernel="linear", probability=True, random_state=seed)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=seed),
        "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "MLP Neural Net": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=seed)),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }

datasets = [
    ("Fruit-Level: DoLP + Reflectance (200 fruits)", fruit_cal, feat_cols_cal, None),
    ("Fruit-Level: Shared-Spot Radial Rings DoLP + Refl (192 fruits)", fruit_v2, vcols, None),
    ("Fruit-Level: Shared-Spot Radial Rings DoLP Only (192 fruits)", fruit_v2, dolp_ring_cols, None),
    ("Spot-Level: DoLP + Reflectance (800 spots, grouped by fruit)", S, feat_cols_cal, S["fruit"].values),
]

rows = []
roc_curves_main = {}
conf_matrices_main = {}

for ds_name, df_ds, cols, groups in datasets:
    X = df_ds[cols].values
    y = df_ds["y"].values
    for m_name in get_models(0).keys():
        accs, baccs, precs, recs, specs, f1s, aucs = [], [], [], [], [], [], []
        y_true_all, y_prob_all, y_pred_all = [], [], []
        for rep in range(5):
            if groups is None:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
                splits = cv.split(X, y)
            else:
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
                splits = cv.split(X, y, groups)
            for tr, te in splits:
                clf = get_models(rep)[m_name]
                clf.fit(X[tr], y[tr])
                yp = clf.predict(X[te])
                pr = clf.predict_proba(X[te])[:, 1]
                accs.append(accuracy_score(y[te], yp))
                baccs.append(balanced_accuracy_score(y[te], yp))
                precs.append(precision_score(y[te], yp, zero_division=0))
                recs.append(recall_score(y[te], yp, zero_division=0))
                tn, fp, fn, tp = confusion_matrix(y[te], yp, labels=[0, 1]).ravel()
                specs.append(tn / max(tn + fp, 1))
                f1s.append(f1_score(y[te], yp, zero_division=0))
                aucs.append(roc_auc_score(y[te], pr))
                if rep == 0 and ds_name == "Fruit-Level: DoLP + Reflectance (200 fruits)":
                    y_true_all.extend(y[te])
                    y_prob_all.extend(pr)
                    y_pred_all.extend(yp)
        if ds_name == "Fruit-Level: DoLP + Reflectance (200 fruits)":
            roc_curves_main[m_name] = roc_curve(y_true_all, y_prob_all)
            conf_matrices_main[m_name] = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        rows.append(dict(
            dataset=ds_name,
            algorithm=m_name,
            n=len(y),
            accuracy=np.mean(accs),
            accuracy_sd=np.std(accs),
            balanced_acc=np.mean(baccs),
            precision=np.mean(precs),
            recall_sensitivity=np.mean(recs),
            specificity=np.mean(specs),
            f1_score=np.mean(f1s),
            AUC=np.mean(aucs),
            AUC_sd=np.std(aucs),
        ))

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(OUT, "ml_algorithms_comparison.csv"), index=False)

pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 220)
for ds_name in res_df.dataset.unique():
    print(f"\n=== {ds_name} ===")
    sub = res_df[res_df.dataset == ds_name].sort_values("AUC", ascending=False)
    print(sub[["algorithm", "accuracy", "accuracy_sd", "precision", "recall_sensitivity", "specificity", "f1_score", "AUC"]].round(3).to_string(index=False))

# Plot 1: Multi-algorithm ROC curves + Metric comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
sub_main = res_df[res_df.dataset == "Fruit-Level: DoLP + Reflectance (200 fruits)"].sort_values("accuracy", ascending=True)

y_pos = np.arange(len(sub_main))
axes[0].barh(y_pos - 0.18, sub_main["accuracy"].values, height=0.34, label="Accuracy", color="#1f77b4", xerr=sub_main["accuracy_sd"].values, capsize=3)
axes[0].barh(y_pos + 0.18, sub_main["AUC"].values, height=0.34, label="ROC-AUC", color="#ff7f0e", xerr=sub_main["AUC_sd"].values, capsize=3)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(sub_main["algorithm"].values)
axes[0].set_xlim(0.75, 1.03)
axes[0].set_xlabel("Score (5x5-Fold Stratified CV)")
axes[0].set_title("A. ML Algorithm Comparison (200 Avocados, DoLP + Reflectance)")
axes[0].legend(loc="lower right")
for i, (acc, auc) in enumerate(zip(sub_main["accuracy"].values, sub_main["AUC"].values)):
    axes[0].text(acc + 0.005, i - 0.18, f"{acc:.3f}", va="center", fontsize=8)
    axes[0].text(auc + 0.005, i + 0.18, f"{auc:.3f}", va="center", fontsize=8)

for m_name in res_df[res_df.dataset == "Fruit-Level: DoLP + Reflectance (200 fruits)"].sort_values("AUC", ascending=False)["algorithm"]:
    fpr, tpr, _ = roc_curves_main[m_name]
    auc_v = res_df[(res_df.dataset == "Fruit-Level: DoLP + Reflectance (200 fruits)") & (res_df.algorithm == m_name)]["AUC"].values[0]
    axes[1].plot(fpr, tpr, lw=2, label=f"{m_name} (AUC = {auc_v:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1].set_ylabel("True Positive Rate (Sensitivity)")
axes[1].set_title("B. Out-of-Fold ROC Curves Across 9 ML Classifiers")
axes[1].legend(loc="lower right", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_ml_algorithms_comparison.png"), dpi=200)

# Plot 2: Confusion matrices for top 6 ML algorithms
top6 = res_df[res_df.dataset == "Fruit-Level: DoLP + Reflectance (200 fruits)"].sort_values("AUC", ascending=False)["algorithm"].values[:6]
fig2, axes2 = plt.subplots(2, 3, figsize=(11, 7))
for ax, m_name in zip(axes2.ravel(), top6):
    cm = conf_matrices_main[m_name]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Immature\n(Pluck Later)", "Mature\n(Pluck Now)"])
    ax.set_yticklabels(["Immature\n(Pluck Later)", "Mature\n(Pluck Now)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (Grower)")
    acc_v = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.set_title(f"{m_name}\n(Acc = {acc_v:.1%})", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > 50 else "black", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_ml_confusion_matrices.png"), dpi=200)
print("\nSaved results/ml_algorithms_comparison.csv, fig_ml_algorithms_comparison.png, and fig_ml_confusion_matrices.png")

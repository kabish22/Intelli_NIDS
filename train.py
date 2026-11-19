#!/usr/bin/env python3
"""Train the intrusion-detection classifier on CIC-IDS2017 (MachineLearningCVE CSVs).

Usage:
    python train.py --data path/to/MachineLearningCVE
    python train.py --data path/to/MachineLearningCVE --binary      # BENIGN vs ATTACK
    python train.py --data ... --skip-compare                        # only train final model
"""
import argparse, glob, json, os, sys, time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


LABEL_CANDIDATES = ["Label", "Attack Type", "Attack_Type", "label", "attack_type"]
BENIGN_NAMES = {"benign", "normal traffic", "normal"}


def load_data(path, binary, min_class):
    """path = folder of CSVs (original CIC-IDS2017) OR a single CSV (e.g. the Kaggle 'cleaned' file)."""
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not files or not all(os.path.isfile(f) for f in files):
        sys.exit(f"No CSV file found at: {path}")
    wanted = set(FEATURES) | set(LABEL_CANDIDATES)
    frames = []
    for f in files:
        print(f"  reading {os.path.basename(f)}")
        df = pd.read_csv(f, encoding="latin1", low_memory=False,
                         usecols=lambda c: c.strip() in wanted)
        df.columns = [c.strip() for c in df.columns]
        label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
        if label_col is None:
            sys.exit(f"{f}: no label column found (looked for {LABEL_CANDIDATES})")
        frames.append(df.rename(columns={label_col: "Label"}))
    df = pd.concat(frames, ignore_index=True)

    feats = [c for c in FEATURES if c in df.columns]
    absent = [c for c in FEATURES if c not in feats]
    if absent:
        print(f"  NOTE: {len(absent)} features are not in this file and will be skipped: {absent}")
    if len(feats) < 10:
        sys.exit(f"Only {len(feats)} usable feature columns found - this file does not look compatible.")
    print(f"  using {len(feats)} features")

    X = (df[feats].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan).fillna(0.0))
    y = (df["Label"].astype(str)
         .str.replace(r"[^\x20-\x7e]", "-", regex=True).str.strip())
    y = y.where(~y.str.lower().isin(BENIGN_NAMES), "BENIGN")

    # Remove exact duplicate rows - otherwise the same flow lands in train AND test.
    data = X.copy()
    data["Label"] = y
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"  dropped {before - len(data):,} duplicate rows -> {len(data):,} remain")
    X, y = data[feats].astype("float32"), data["Label"]

    if binary:
        y = y.where(y == "BENIGN", "ATTACK")
    counts = y.value_counts()
    keep = counts[counts >= min_class].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"  dropping classes with < {min_class} samples: {dropped}")
    mask = y.isin(keep)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), feats


def candidate_models():
    pre = lambda: [FunctionTransformer(signed_log), StandardScaler()]
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=25, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=60, min_samples_leaf=2, max_depth=30,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "KNN (k=5)": make_pipeline(*pre(), KNeighborsClassifier(5, n_jobs=-1)),
        "Logistic Regression": make_pipeline(*pre(), LogisticRegression(max_iter=300, class_weight="balanced")),
        "MLP (64-32)": make_pipeline(*pre(), MLPClassifier((64, 32), max_iter=60, early_stopping=True, random_state=42)),
    }


def compare(Xtr, ytr, Xte, yte, out_dir, n_train=100_000, n_test=30_000):
    print("\n== Classifier comparison (subsampled) ==")
    rng = np.random.RandomState(42)
    tr = rng.choice(len(Xtr), min(n_train, len(Xtr)), replace=False)
    te = rng.choice(len(Xte), min(n_test, len(Xte)), replace=False)
    rows = []
    for name, model in candidate_models().items():
        t0 = time.time()
        model.fit(Xtr.iloc[tr], ytr.iloc[tr])
        fit_s = time.time() - t0
        t0 = time.time()
        pred = model.predict(Xte.iloc[te])
        pred_s = time.time() - t0
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yte.iloc[te], pred),
            "macro_f1": f1_score(yte.iloc[te], pred, average="macro"),
            "weighted_f1": f1_score(yte.iloc[te], pred, average="weighted"),
            "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 1),
        })
        print(f"  {name:22s} acc={rows[-1]['accuracy']:.4f} macroF1={rows[-1]['macro_f1']:.4f} "
              f"fit={fit_s:.1f}s")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)


def plot_confusion(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), max(5, len(classes) * 0.7)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalised confusion matrix")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(model, feats, path):
    imp = pd.Series(model.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 7))
    imp.plot.barh(ax=ax); ax.set_title("Random Forest feature importance")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of CIC-IDS2017 CSVs, or a single CSV file")
    ap.add_argument("--out", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--binary", action="store_true", help="BENIGN vs ATTACK instead of attack types")
    ap.add_argument("--min-class", type=int, default=100)
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument("--trees", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    print("Loading data...")
    X, y, feats = load_data(args.data, args.binary, args.min_class)
    print("\nClass distribution:\n" + y.value_counts().to_string())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    if not args.skip_compare:
        compare(Xtr, ytr, Xte, yte, args.reports)

    print("\n== Training final Random Forest on the full training split ==")
    rf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=2, max_depth=30,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    t0 = time.time()
    rf.fit(Xtr, ytr)
    print(f"  trained in {time.time() - t0:.1f}s")

    pred = rf.predict(Xte)
    report = classification_report(yte, pred, digits=4, zero_division=0)
    print("\n" + report)
    with open(os.path.join(args.reports, "classification_report.txt"), "w") as fh:
        fh.write(report)
    plot_confusion(yte, pred, list(rf.classes_), os.path.join(args.reports, "confusion_matrix.png"))
    plot_importance(rf, feats, os.path.join(args.reports, "feature_importance.png"))

    bundle = {"model": rf, "features": feats, "benign_label": "BENIGN",
              "binary": args.binary, "classes": list(rf.classes_)}
    path = os.path.join(args.out, "nids_rf.joblib")
    joblib.dump(bundle, path, compress=3)
    print(f"\nSaved model -> {path}\nReports  -> {args.reports}/")


if __name__ == "__main__":
    main()

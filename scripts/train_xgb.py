from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import json

# create results dir
ROOT = Path(__file__).resolve().parents[1]  # repo root
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

best_iteration = -1
bst_booster = None

try:
    df = pd.read_csv('data_preprocessing/data/processed/preprocessed_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Preprocessed CSV not found. Check your path.")

# prepare X/y
y = df['default'].astype(int)
X = df.drop(columns=['default'])

# save preprocessing metadata and feature list
feature_names = X.columns.to_list()
num_cols = X.select_dtypes(include='number').columns.tolist()
preproc = {
    "feature_names": feature_names,
    "num_cols": num_cols,
    "median": X[feature_names].median().to_dict()
}
joblib.dump(preproc, MODELS / "preproc.joblib")


# train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# class imbalance baseline
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / max(1, pos)
print(f"neg/pos = {neg}/{pos}, scale_pos_weight={scale_pos_weight:.2f}")


model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# Option A: sklearn-style with early_stopping_rounds
try:
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=50)
    best_iteration = int(getattr(model, "best_iteration", -1))
except TypeError as e:
    print("sklearn fit() does not accept early_stopping_rounds:", e)
    # Option B: callbacks API
    try:
        cb = xgb.callback.EarlyStopping(rounds=50, save_best=True)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[cb])
        best_iteration = int(getattr(model, "best_iteration", -1))
    except Exception:
        # Option C: fallback to xgb.train (always works)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = model.get_xgb_params() if hasattr(model, "get_xgb_params") else model.get_params()
        params = {k: v for k, v in params.items() if k != "n_estimators"}
        num_round = int(model.get_params().get("n_estimators", 100))
        bst_booster = xgb.train(params, dtrain, num_boost_round=num_round,
                                evals=[(dtest, "eval")], early_stopping_rounds=50, verbose_eval=50)
        best_iteration = int(getattr(bst_booster, "best_iteration", -1))


# save metrics (use best_iteration local variable)
metrics = {
    "best_iteration": int(best_iteration)
}
with open(RESULTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save model or booster depending on training method
if bst_booster is not None:
    bst_booster.save_model(str(MODELS / "xgb_baseline_booster.model"))
    print("Saved booster to", MODELS / "xgb_baseline_booster.model")
else:
    joblib.dump(model, MODELS / "xgb_baseline.joblib")
    print("Saved sklearn wrapper model to", MODELS / "xgb_baseline.joblib")

# Prediction: always respect best_iteration
def _get_booster(m):
    try:
        return m.get_booster()
    except Exception:
        return m  # maybe already a booster

booster = bst_booster if bst_booster is not None else _get_booster(model)
dtest = xgb.DMatrix(X_test)
if best_iteration and best_iteration > 0:
    try:
        y_proba = booster.predict(dtest, iteration_range=(0, best_iteration + 1))
    except TypeError:
        y_proba = booster.predict(dtest, ntree_limit=best_iteration)
else:
    # fallback
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = booster.predict(dtest)
y_pred = (np.array(y_proba) > 0.5).astype(int)


print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

metrics = {
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "scale_pos_weight": scale_pos_weight,
    "best_iteration": int(getattr(model, "best_iteration", -1))
}
with open(RESULTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save model
joblib.dump(model, MODELS / "xgb_baseline.joblib")
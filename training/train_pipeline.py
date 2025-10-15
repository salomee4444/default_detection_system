# training/train_pipeline.py
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from spec_encoder import SpecEncoder  # uses your encoder

# ---------- Paths ----------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
DATA_DIR = PROJECT_ROOT / "data"          # CSVs live here (gitignored)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = ARTIFACT_DIR / "model_pipeline.joblib"

ENCODER_SPEC_PATH = PROJECT_ROOT / "encoder_suggestions.txt"
TARGET_COL = "TARGET"
THRESHOLD = float(0.30)

def _find_csv(patterns):
    files = []
    for p in patterns:
        files.extend(DATA_DIR.glob(p))
    return sorted(files)

def load_train_valid():
    """
    Loads the cleaned, merged dataset for training.
    Splits into 80/20 train/validation sets (stratified on TARGET).
    """
    DATA_FILE = PROJECT_ROOT / "data" / "merged_df_clean.csv"
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows from {DATA_FILE.name}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column not found in {DATA_FILE.name}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.assign(TARGET=y_train.values)
    valid_df = X_valid.assign(TARGET=y_valid.values)
    return train_df, valid_df

def make_pipeline(enc_spec: dict) -> Pipeline:
    encoder = SpecEncoder(
        spec=enc_spec,
        target_col=TARGET_COL,
        exclude_cols=["SK_ID_CURR", "SK_ID_PREV", "community_id"]
    )
    return Pipeline(steps=[
        ("encoder", encoder),                          # returns a numeric DataFrame
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

def main():
    # --- Load data ---
    train_df, valid_df = load_train_valid()

    # --- Split X/y ---
    y_train = train_df[TARGET_COL].astype(int).values
    X_train = train_df.drop(columns=[TARGET_COL])
    y_valid = valid_df[TARGET_COL].astype(int).values
    X_valid = valid_df.drop(columns=[TARGET_COL])

    # --- Encoder spec ---
    enc_spec = json.loads(ENCODER_SPEC_PATH.read_text(encoding="utf-8"))

    # --- Build pipeline ---
    pipe = make_pipeline(enc_spec)

    # IMPORTANT:
    # SpecEncoder.fit expects TARGET to be present in the DataFrame it sees during .fit().
    # We pass X_train augmented with TARGET so the encoder can compute target/onehot stats.
    pipe.fit(X_train.assign(TARGET=y_train), y_train)

    # --- Evaluate ---
    y_prob = pipe.predict_proba(X_valid)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("AUC:", roc_auc_score(y_valid, y_prob).round(4))
    print(classification_report(y_valid, y_pred, digits=4))

    # --- Persist artifact ---
    import joblib
    joblib.dump(pipe, ARTIFACT_PATH)
    print(f"✅ Saved pipeline to {ARTIFACT_PATH.resolve()}")

if __name__ == "__main__":
    main()
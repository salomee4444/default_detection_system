# training/spec_encoder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_MISSING_TOKEN = "__MISSING__"


class SpecEncoder(BaseEstimator, TransformerMixin):
    """
    Encode a DataFrame according to a provided spec:
      - 'none'   -> pass through numeric columns unchanged
      - 'binary' -> ensure columns are 0/1 (object two-level mapped to 0/1; numeric two-level coerced to {0,1})
      - 'onehot' -> one-hot encode using categories seen at fit; missing/unseen -> _MISSING_TOKEN bucket
      - 'target' -> mean target encoding with smoothing; unseen -> global mean

    Notes
    -----
    - This transformer is sklearn-compatible. During Pipeline.fit(X, y) sklearn
      may call fit_transform(X, y). Both fit(...) and fit_transform(...)
      accept an optional y and, if TARGET is absent in X but y is provided,
      they will temporarily stitch y into the dataframe under `target_col`.
    - Use `exclude_cols` to drop identifiers like ["SK_ID_CURR","SK_ID_PREV","community_id"].
    """

    def __init__(
        self,
        spec: Dict[str, Dict[str, Any]],
        target_col: str = "TARGET",
        smoothing: float = 20.0,
        exclude_cols: Optional[List[str]] = None,
    ):
        self.spec = spec
        self.target_col = target_col
        self.smoothing = float(smoothing)
        self.exclude_cols = set(exclude_cols or [])

        # Fitted state
        self.binary_maps_: Dict[str, Dict[Any, int]] = {}
        self.target_stats_: Dict[str, Dict[str, Any]] = {}   # col -> {'mapping': {cat: enc}, 'global_mean': float}
        self.onehot_levels_: Dict[str, pd.Index] = {}        # col -> categories seen at fit
        self.ohe_columns_: List[str] = []                    # final onehot column names (sorted)
        self.passthrough_cols_: List[str] = []               # 'none'/'binary'/'target' final column names
        self.fitted_ = False

    # ---------- binary helpers ----------
    @staticmethod
    def _to_binary_mapping(series: pd.Series) -> Dict[Any, int]:
        """Create a stable 0/1 mapping for a binary series (object or numeric)."""
        uniq = pd.Series(series.dropna().unique())

        if len(uniq) <= 1:
            return {uniq.iloc[0]: 0} if len(uniq) == 1 else {}

        if pd.api.types.is_numeric_dtype(series):
            s = sorted(uniq.tolist())
            return {s[0]: 0, s[1]: 1}

        lowered = {str(v).strip().lower() for v in uniq}
        if lowered.issubset({"y", "n"}):
            return {v: (0 if str(v).strip().lower() == "n" else 1) for v in uniq}
        if lowered.issubset({"yes", "no"}):
            return {v: (0 if str(v).strip().lower() == "no" else 1) for v in uniq}
        if lowered.issubset({"m", "f"}):
            return {v: (0 if str(v).strip().lower() == "f" else 1) for v in uniq}

        s = sorted(uniq.astype(str).tolist())
        return {v: (0 if str(v) == s[0] else 1) for v in uniq}

    # ---------- target encoding ----------
    def _fit_target_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Mean target encoding with smoothing; missing treated as a category."""
        y = df[self.target_col].astype(float)
        x = df[col].astype("object").fillna(_MISSING_TOKEN)

        global_mean = y.mean()
        counts = x.value_counts()
        means = y.groupby(x).mean()

        enc = (means * counts + self.smoothing * global_mean) / (counts + self.smoothing)
        mapping = enc.to_dict()

        self.target_stats_[col] = {
            "mapping": mapping,
            "global_mean": float(global_mean),
        }

    def _apply_target_encoder(self, s: pd.Series, col: str) -> pd.Series:
        st = self.target_stats_[col]
        x = s.astype("object").fillna(_MISSING_TOKEN)
        return x.map(st["mapping"]).fillna(st["global_mean"]).astype(float)

    # ---------- core API ----------
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "SpecEncoder":
        """
        Fit the encoder on a DataFrame. If `target_col` is not present and `y`
        is provided, a temporary TARGET column is added for fitting.
        """
        # If TARGET missing but y provided, stitch it in for fitting
        if self.target_col not in df.columns and y is not None:
            df = df.assign(**{self.target_col: pd.Series(y, index=df.index)})

        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in DataFrame.")

        # Reset fitted state
        self.binary_maps_.clear()
        self.target_stats_.clear()
        self.onehot_levels_.clear()
        self.ohe_columns_.clear()
        self.passthrough_cols_.clear()

        # Fit per-column logic
        for col, meta in self.spec.items():
            if col == self.target_col or col in self.exclude_cols:
                continue
            if col not in df.columns:
                continue

            enc = meta.get("encoder", "none")

            if enc == "binary":
                mapping = self._to_binary_mapping(df[col])
                self.binary_maps_[col] = mapping

            elif enc == "target":
                self._fit_target_encoder(df, col)

            elif enc == "onehot":
                s = df[col].astype("object").fillna(_MISSING_TOKEN)
                cats = pd.Index(sorted(s.astype(str).unique()))
                self.onehot_levels_[col] = cats

        # Build stable OHE column list
        if self.onehot_levels_:
            tmp = []
            for c, cats in self.onehot_levels_.items():
                for cat in cats:
                    tmp.append(f"{c}__{cat}")
            self.ohe_columns_ = sorted(tmp)

        # Passthrough order (keep for stable column ordering)
        passthrough = []
        for col, meta in self.spec.items():
            if col == self.target_col or col in self.exclude_cols or col not in df.columns:
                continue
            enc = meta.get("encoder", "none")
            if enc in ("none", "binary", "target"):
                passthrough.append(col)
        self.passthrough_cols_ = sorted(passthrough)

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("SpecEncoder is not fitted. Call .fit(...) first.")

        out: Dict[str, pd.Series] = {}

        # 1) 'none' -> passthrough numeric
        for col, meta in self.spec.items():
            if col in self.exclude_cols or col == self.target_col or col not in df.columns:
                continue
            if meta.get("encoder", "none") == "none":
                out[col] = pd.to_numeric(df[col], errors="coerce")

        # 2) 'binary' -> map via fitted mapping (fallbacks for unseen/missing)
        for col, mapping in self.binary_maps_.items():
            if col not in df.columns or col in self.exclude_cols:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                mapped = s.map(mapping)
                if mapped.isna().any():
                    mapped = (s.fillna(0) != 0).astype(int)
            else:
                mapped = s.map(mapping)
                if mapped.isna().any():
                    mapped = mapped.fillna(0).astype(int)
            out[col] = mapped.astype(int)

        # 3) 'target' -> apply mean encoding
        for col in self.target_stats_.keys():
            if col in df.columns and col not in self.exclude_cols:
                out[col] = self._apply_target_encoder(df[col], col)

        # 4) 'onehot' -> fixed OHE columns; unseen -> _MISSING_TOKEN bucket
        for col, cats in self.onehot_levels_.items():
            if col in self.exclude_cols:
                for cat in cats:
                    out[f"{col}__{cat}"] = pd.Series(0, index=df.index, dtype=int)
                continue

            if col not in df.columns:
                for cat in cats:
                    out[f"{col}__{cat}"] = pd.Series(0, index=df.index, dtype=int)
                continue

            s = df[col].astype("object").fillna(_MISSING_TOKEN)
            valid_set = set(cats)
            s = s.where(s.isin(valid_set), _MISSING_TOKEN)

            dummies = pd.get_dummies(s, prefix=col)
            dummies.columns = [c.replace(f"{col}_", f"{col}__") for c in dummies.columns]

            # ensure all fitted columns exist
            for cat in cats:
                cname = f"{col}__{cat}"
                if cname not in dummies.columns:
                    dummies[cname] = 0

            dummies = dummies[[f"{col}__{cat}" for cat in cats]].astype(int)

            for cname in dummies.columns:
                out[cname] = dummies[cname]

        # Assemble final frame in a stable order
        col_order: List[str] = []
        col_order.extend(self.passthrough_cols_)
        col_order.extend([c for c in self.ohe_columns_ if c in out])

        remaining = [c for c in out.keys() if c not in col_order]
        col_order.extend(sorted(remaining))

        X = pd.DataFrame(out, index=df.index)
        X = X.reindex(columns=col_order, fill_value=0)
        return X

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        sklearn-compatible fit_transform. Accepts optional `y`.
        If TARGET is missing but `y` is provided, stitches `y` into df for fitting.
        """
        if self.target_col not in df.columns and y is not None:
            df_with_target = df.assign(**{self.target_col: pd.Series(y, index=df.index)})
        else:
            df_with_target = df
        self.fit(df_with_target, y=None)  # y already injected if needed
        return self.transform(df)

    # ---------- persistence & metadata ----------
    def save(self, path: str) -> None:
        state = {
            "spec": self.spec,
            "target_col": self.target_col,
            "smoothing": self.smoothing,
            "binary_maps_": self.binary_maps_,
            "target_stats_": self.target_stats_,
            "onehot_levels_": {k: list(v) for k, v in self.onehot_levels_.items()},
            "ohe_columns_": self.ohe_columns_,
            "passthrough_cols_": self.passthrough_cols_,
            "exclude_cols": list(self.exclude_cols),
            "fitted_": self.fitted_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "SpecEncoder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            state["spec"],
            target_col=state["target_col"],
            smoothing=state["smoothing"],
            exclude_cols=state.get("exclude_cols", []),
        )
        obj.binary_maps_ = state["binary_maps_"]
        obj.target_stats_ = state["target_stats_"]
        obj.onehot_levels_ = {k: pd.Index(v) for k, v in state["onehot_levels_"].items()}
        obj.ohe_columns_ = state["ohe_columns_"]
        obj.passthrough_cols_ = state["passthrough_cols_"]
        obj.fitted_ = state["fitted_"]
        return obj

    @property
    def feature_names_(self) -> List[str]:
        if not self.fitted_:
            raise RuntimeError("SpecEncoder is not fitted.")
        return list(self.passthrough_cols_) + list(self.ohe_columns_)
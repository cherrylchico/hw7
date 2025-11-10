# Create at least two feature classes that transform some of the columns in the data set. These
# feature classes need to have the same structure defined by an abstract parent class (Remember:
# polymorphism).

import pandas as pd
import numpy as np
from typing import Iterable
from ..interfaces.base import FeatureEngineerInterface

class CategoricalEncoder(FeatureEngineerInterface):
    """
    Encode categorical features.
    Supported methods:
    - 'onehot' : simple pd.get_dummies (no fit required)
    - 'target' : target encoding (requires fit with target provided)
    """
    
    def __init__(self, cols: Iterable[str] | None = None, method: str = "onehot"):
        if method not in {"onehot", "target"}:
            raise ValueError(f"Unsupported method: {method}")
        
        self.cols = list(cols) if cols is not None else None
        self.method = method
        self.target_means: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame, target: pd.Series | None = None) -> "CategoricalEncoder":
        self._auto_detect_cols(df)
        
        if self.method == "onehot":
            self._fit_onehot(df, target)
        elif self.method == "target":
            self._fit_target(df, target)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._auto_detect_cols(df)
        
        if self.method == "onehot":
            return self._transform_onehot(df)
        elif self.method == "target":
            return self._transform_target(df)


    def _auto_detect_cols(self, df: pd.DataFrame) -> None:
        if self.cols is None:
            self.cols = df.select_dtypes(include="object").columns.tolist()
    
    def _fit_onehot(self, df: pd.DataFrame, target: pd.Series | None = None) -> None:
        pass
    
    def _fit_target(self, df: pd.DataFrame, target: pd.Series | None = None) -> None:
        if target is None:
            raise ValueError("target must be provided for target encoding")
        
        combined = df.copy()
        combined["_target"] = target.values
        
        for col in self.cols:
            if col in combined.columns:
                self.target_means[col] = combined.groupby(col)["_target"].mean().to_dict()
    

    
    def _transform_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_encode = [c for c in self.cols if c in df.columns]
        
        if not cols_to_encode:
            return df
        
        dummies = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode, drop_first=True)
        return pd.concat([df.drop(columns=cols_to_encode), dummies], axis=1)
    
    def _transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols_to_drop = []
        
        for col, mapping in self.target_means.items():
            if col in out.columns:
                default_value = pd.Series(mapping).mean()
                out[f"{col}_te"] = out[col].map(mapping).fillna(default_value)
                cols_to_drop.append(col)  # Mark for removal
        
        # Drop original categorical columns
        out = out.drop(columns=cols_to_drop)
        return out
    

        

class FeatureScaler(FeatureEngineerInterface):
    """
    Scale numerical features using standardization (z-score normalization).
    Formula: (x - mean) / std
    """
    
    def __init__(self, cols: Iterable[str] | None = None):
        self.cols = list(cols) if cols is not None else None
        self.scaling_params: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame, target: pd.Series | None = None) -> "FeatureScaler":
        self._auto_detect_cols(df)
        
        for col in self.cols:
            if col in df.columns:
                self.scaling_params[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std()
                }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._auto_detect_cols(df)
        out = df.copy()
        
        for col, params in self.scaling_params.items():
            if col in out.columns:
                out[col] = (out[col] - params["mean"]) / params["std"]
        
        return out
    
    def _auto_detect_cols(self, df: pd.DataFrame) -> None:
        if self.cols is None:
            self.cols = df.select_dtypes(include=[np.number]).columns.tolist()
    

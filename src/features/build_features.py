
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import joblib
from pathlib import Path 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,    
    StandardScaler,
    OrdinalEncoder

)

from src.utils.logger import get_logger

from src.utils.config import load_config


logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        self.cfg      = load_config(config_path)
        self.target   = self.cfg["data"]["target_column"]
        self.paths_cfg = self.cfg["paths"]
        self.scale = self.cfg["feature_engineering"].get("scale", False)

        self._scaler: Optional[object]           = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scale_cols: List[str]              = []
        self._engineered_cols: List[str]         = []
        self.feature_names_out_: List[str]       = []



    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        
        logger.info("=== Feature Engineering FIT ===")
        X = X.copy()

        # 2. Encode categoricals
        X = self._encode_categoricals(X, fit=True)
        
        # 3. Scaling
        if self.scale:
            self._fit_scaler(X)

        self.feature_names_out_ = X.columns.tolist()
        logger.info(f"Feature engineering fitted. Output features: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        
        logger.info("=== Feature Engineering TRANSFORM ===")
        X = X.copy()

        X = self._encode_categoricals(X, fit=False)
        
        if self.scale:
            X = self._apply_scaler(X)

        # Reorder to match fitted order
        for col in self.feature_names_out_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[[c for c in self.feature_names_out_ if c in X.columns]]
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)
    
    
    ## -------------------- Encoding ---------------------
    

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return X

        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique == 2:
                # Binary → label encode
                if fit:
                    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1 )
                    X[col] = oe.fit_transform(X[[col]].astype(str)).ravel()
                    self._label_encoders[col] = oe
                else:
                    oe = self._label_encoders.get(col)
                    if oe:
                       X[col] = oe.transform(X[[col]].astype(str)).ravel()
            elif n_unique <= 15:
                # Low cardinality → one-hot encode
                if fit:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                    self._label_encoders[f"_ohe_{col}"] = dummies.columns.tolist()
                else:
                    expected_cols = self._label_encoders.get(f"_ohe_{col}", [])
                    dummies       = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    for ec in expected_cols:
                        if ec not in dummies.columns:
                            dummies[ec] = 0
                    X = pd.concat([X.drop(columns=[col]), dummies[expected_cols]], axis=1)
            else:
                # High cardinality → frequency encoding
                if fit:
                    freq = X[col].value_counts(normalize=True).to_dict()
                    self._label_encoders[f"_freq_{col}"] = freq
                else:
                    freq = self._label_encoders.get(f"_freq_{col}", {})
                X[col] = X[col].map(freq if fit else self._label_encoders.get(f"_freq_{col}", {})).fillna(0)

        logger.info(f"Categorical encoding applied to {len(cat_cols)} column(s).")
        return X


    ## ------------------------ Scaling ------------------------------------

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        
        scale_cols = X.select_dtypes(include=np.number  ).columns.tolist()
    
        if not scale_cols:
            logger.info("No numerical columns found for scaling.")
            return
    
        self._scale_cols = scale_cols
    
        self._scaler = StandardScaler()
        self._scaler.fit(X[self._scale_cols])
    
        logger.info(
            "StandardScaler fitted on %d numerical feature(s): %s",
            len(self._scale_cols),
            self._scale_cols
        )

    
    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or not self._scale_cols:
            return X
    
        X = X.copy()
        X[self._scale_cols] = self._scaler.transform(X[self._scale_cols])
    
        return X
    
    

    ## ------------------ Persistence -----------------------

    def save_scaler(self, path: str = "data/processed/scaler.joblib") -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        logger.info(f"Scaler saved → {path}")

    def save_feature_names(self, path: str = "artifacts/feature_engineering/feature_names.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.feature_names_out_, f, indent=2)
        logger.info(f"Feature names saved → {path}")
        
    def save_feature_engineer(self, path: str = "artifacts/feature_engineering/feature_engineer.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer saved → {path}") 
    
    
    def save_parquets(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> None:
    
        if self.scale:
            x_paths = {
                "train": self.paths_cfg["processed_train_scale"],
                "val": self.paths_cfg["processed_val_scale"],
                "test": self.paths_cfg["processed_test_scale"],
            }
            logger.info("Saving SCALED datasets.")
        else:
            x_paths = {
                "train": self.paths_cfg["processed_train"],
                "val": self.paths_cfg["processed_val"],
                "test": self.paths_cfg["processed_test"],
            }
            logger.info("Saving UNSCALED datasets.")
    
        datasets = {
            x_paths["train"]: X_train,
            x_paths["val"]: X_val,
            x_paths["test"]: X_test,
            self.paths_cfg["processed_y_train"]: y_train,
            self.paths_cfg["processed_y_val"]: y_val,
            self.paths_cfg["processed_y_test"]: y_test,
        }
    
        for file_path, obj in datasets.items():
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
    
            if isinstance(obj, pd.DataFrame):
                obj.to_parquet(path, index=False)
    
                logger.info(
                    f"Saved {path.resolve()} | shape={obj.shape}"
                )
    
            elif isinstance(obj, np.ndarray):
                df = pd.DataFrame(
                    obj,
                    columns=self.selected_features_
                )
    
                df.to_parquet(path, index=False)
    
                logger.info(
                    f"Saved {path.resolve()} | shape={df.shape}"
                )
    
            else:
                df = pd.Series(
                    obj,
                    name=self.target
                ).to_frame()
    
                df.to_parquet(path, index=False)
    
                logger.info(
                    f"Saved {path.resolve()} | shape={df.shape}"
                )
    
            
            
            
    
                
            
            
            
                 
    
    
    
    

    

    
        

    
       
            
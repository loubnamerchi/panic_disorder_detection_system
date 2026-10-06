
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
    OrdinalEncoder,
    OneHotEncoder

)

from src.utils.logger import get_logger

from src.utils.config import load_config


logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg      = load_config(config_path)
       
        self.categorical_columns_ = []
        self._ordinal_encoders = {}
        self._onehot_encoder = None

        self._scaler: Optional[object]           = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scale_cols: List[str]              = []
        self._engineered_cols: List[str]         = []
        self.feature_names_out_: List[str]       = []

    
    
    def encode_ordinal(self,
        X: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
    
        X = X.copy()
    
        cat_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    
        for col in cat_cols:
    
            if fit:
    
                oe = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
    
                X[col] = oe.fit_transform(
                    X[[col]].astype(str)
                ).ravel()
    
                self._ordinal_encoders[col] = oe
    
            else:
    
                oe = self._ordinal_encoders[col]
    
                X[col] = oe.transform(
                    X[[col]].astype(str)
                ).ravel()
    
        return X

    def encode_onehot(self,
        X: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
    
        X = X.copy()
    
        # ---------------------------------------------------------
        # Identify categorical columns from training data
        # ---------------------------------------------------------
        if fit:
            self.categorical_columns_ = (
                X.select_dtypes(
                    include=["object", "category"]
                )
                .columns
                .tolist()
            )
    
        cat_cols = [
            col
            for col in self.categorical_columns_
            if col in X.columns
        ]
    
        if not cat_cols:
            return X
    
        # ---------------------------------------------------------
        # Fit OneHotEncoder
        # ---------------------------------------------------------
        if fit:
    
            self._onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
                dtype=int,
            )
    
            encoded = self._onehot_encoder.fit_transform(
                X[cat_cols]
            )
    
            encoded_columns = (
                self._onehot_encoder.get_feature_names_out(cat_cols)
            )
    
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoded_columns,
                index=X.index,
            )
    
            X = pd.concat(
                [
                    X.drop(columns=cat_cols),
                    encoded_df,
                ],
                axis=1,
            )
    
        else:
    
            if self._onehot_encoder is None:
                raise RuntimeError(
                    "OneHotEncoder has not been fitted."
                )
    
            encoded = self._onehot_encoder.transform(
                X[cat_cols]
            )
    
            encoded_columns = (
                self._onehot_encoder.get_feature_names_out(cat_cols)
            )
    
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoded_columns,
                index=X.index,
            )
    
            X = pd.concat(
                [
                    X.drop(columns=cat_cols),
                    encoded_df,
                ],
                axis=1,
            )
    
        return X
    

    ## ------------------ Persistence -----------------------
        
    def save_feature_engineer(self, path: str = "artifacts/feature_engineering/feature_engineer.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer saved → {path}") 
    
    

            
            
                 
    
    
    
    

    

    
        

    
       
            
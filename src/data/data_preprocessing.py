

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pathlib
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger 

from src.utils.config import load_config


logger = get_logger(__name__)


class DataPreprocessor:

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        self.cfg      = load_config(config_path)
        self.target   = self.cfg["data"]["target_column"]
        self.paths_cfg = self.cfg["paths"]
        
        self._imputer = None
        self._cat_modes = {}
        
        self.cleaning_report: Dict = {}


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
      
        logger.info("=== Data Preprocessing START ===")
        original_shape = df.shape
        df = df.copy()

        df = self._handle_missing(df, fit=True)

        self.cleaning_report = {
            "original_shape": original_shape,
            "clean_shape":    df.shape,
            "rows_removed":   original_shape[0] - df.shape[0],
        }
        logger.info(
            f"Preprocessing complete. "
            f"{original_shape[0]:,} → {df.shape[0]:,} rows "
            f"({self.cleaning_report['rows_removed']:,} removed)"
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        df = self._handle_missing(df, fit=False)
        return df
 
    ## ---------------- Handle missing values -------------------

    
    def _handle_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        null_counts = df.isnull().sum()
        total_missing = null_counts.sum()
    
        if total_missing == 0:
            logger.info("No missing values — skipping imputation.")
            return df
    
        logger.info("Handling %s missing values ""(numerical: median, categorical: mode)",f"{total_missing:,}" )
    
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(
            include=["object", "category","bool"]
        ).columns.tolist()
    
        # Numerical columns → Median
        if num_cols:
            if fit:
                self._imputer = SimpleImputer(strategy="median")
                df[num_cols] = self._imputer.fit_transform(df[num_cols])
            else:
                if self._imputer is None:
                    raise RuntimeError("Imputer not fitted — call fit_transform first.")
    
                df[num_cols] = self._imputer.transform(df[num_cols])
    
        # Categorical columns → Mode
        if cat_cols:
            if fit:
                self._cat_modes = {}
    
                for col in cat_cols:
                    mode = df[col].mode()
    
                    if not mode.empty:
                        self._cat_modes[col] = mode.iloc[0]
                    else:
                        self._cat_modes[col] = "UNKNOWN"
    
                    df[col] = df[col].fillna(self._cat_modes[col])
    
            else:
                if not hasattr(self, "_cat_modes"):
                    raise RuntimeError("Categorical imputation values not fitted.")
    
                for col in cat_cols:
                    if col in self._cat_modes:
                        df[col] = df[col].fillna(self._cat_modes[col])
    
        return df
    
    
    ## ------------------- Persistence -------------------
        
    def save_preprocessor(self,path: str = "artifacts/preprocessing/preprocessor.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"DataPreprocessor saved → {path}")
        
    
    def save_parquets(
            self,
            X_train: pd.DataFrame,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_val: pd.Series,
            y_test: pd.Series,
        ) -> None:
        
            
            x_paths = {
                "train": self.paths_cfg["processed_train"],
                "val": self.paths_cfg["processed_val"],
                "test": self.paths_cfg["processed_test"],
            }
            logger.info("Saving PreProcessed datasets.")
       
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
            
    
        
    

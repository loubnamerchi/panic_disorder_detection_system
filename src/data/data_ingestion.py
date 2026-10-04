
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import  Optional

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)



class DataIngestion:

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg      = load_config(config_path)       
        self.data_cfg = self.cfg["data"]
        self.target   = self.data_cfg["target_column"]

    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        raw_path = Path(path or self.data_cfg["raw_path"])
        logger.info(f"Loading data from: {raw_path.resolve()}")

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path.resolve()}")

        suffix = raw_path.suffix.lower()
        readers = {
            ".csv":     pd.read_csv,
            ".parquet": pd.read_parquet,
            ".xlsx":    pd.read_excel,
            ".xls":     pd.read_excel,
        }
        if suffix not in readers:
            raise ValueError(f"Unsupported file type: {suffix}")

        df = readers[suffix](raw_path)
        logger.info(f"Loaded  shape  : {df.shape}")
        logger.info(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        if self.target in df.columns:
            posted_rate = df[self.target].mean()
            logger.info(f"Target '{self.target}' — Mean: {posted_rate:,.2f}")

        return df

    def print_overview(self, df: pd.DataFrame) -> None:
        sep = "─" * 60
    
        print(f"\n{sep}")
        print("  DATASET OVERVIEW")
        print(sep)
    
        # ── Basic information ──────────────────────────────────────
        print(f"  Shape      : {df.shape}")
        print(
            f"  Memory     : "
            f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB"
        )
    
        # ── Data types ─────────────────────────────────────────────
        print(f"\n  Dtypes summary:")
        print(
            df.dtypes
            .value_counts()
            .to_string(header=False)
        )
    
        # ── Missing values ─────────────────────────────────────────
        print(f"\n  Missing values:")
    
        null_info = df.isnull().sum()
        null_info = null_info[null_info > 0]
    
        if null_info.empty:
            print("    → None ✓")
        else:
            print(null_info.to_string())
    
        # ── Duplicates ─────────────────────────────────────────────
        print(
            f"\n  Duplicated rows: "
            f"{df.duplicated().sum():,}"
        )
    
        # ── Target ─────────────────────────────────────────────────
        if self.target in df.columns:
    
            target = df[self.target]
    
            print(
                f"\n  Target ('{self.target}'):"
            )
    
            print(
                f"    Type   : {target.dtype}"
            )
    
            print(
                f"    Missing: {target.isna().sum():,}"
            )
    
        # ── Descriptive statistics ─────────────────────────────────
        print(f"\n  Descriptive stats (numeric):")
        print(df.describe().to_string())
    
        print(f"{sep}\n")

    

from __future__ import annotations

import time

from sklearn.model_selection import train_test_split

from src.data.data_ingestion     import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.build_features  import FeatureEngineer
from src.utils.logger             import get_logger
from src.utils.config             import load_config
from pathlib import Path
import yaml

logger = get_logger(__name__)


def run_pipeline(config_path: str = "config.yaml") -> dict:
    t0  = time.time()
    cfg = load_config(config_path)
    OUTPUT_PATH = Path("categorical_values.yaml")

    # ── 1. Ingest ────────────────────────────────────────────────
    logger.info("\n[1/4] DATA INGESTION")
    ingestor   = DataIngestion(config_path)
    df_raw     = ingestor.load()
    ingestor.print_overview(df_raw)
    
    # --- 1.1 Save categorical values into .yaml
    categorical_columns = df_raw.select_dtypes(include=["object", "category"]).columns
    categorical_values = {}
    for column in categorical_columns:
        values = (
            df_raw[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        # Sort for consistent YAML output
        values.sort()
        categorical_values[column] = values
        
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
        categorical_values,
        file,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    # ── 2. SPLIT FIRST — before ANY fitting ──────────────────────
    logger.info("\n[2/4] SPLIT FIRST (before fitting anything)")
    target    = cfg["data"]["target_column"]
    rs        = cfg["training"]["random_state"]
    id    = cfg["data"]["id_column"]

    X_raw = df_raw.drop(columns=[target,id], errors="ignore")
    y_raw = df_raw[target]

    test_size = cfg["split"]["test_size"]
    val_size  = cfg["split"]["val_size"]
    stratify  = y_raw if cfg["split"].get("stratify", True) else None

    # First cut: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_raw,
        test_size=test_size,
        random_state=rs,
        stratify=stratify
    )
    # Second cut: validation from remaining
    val_ratio = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=rs,
        stratify=y_temp if stratify is not None else None
    )

    logger.info(f"Train: {X_train_raw.shape} | "
                f"Val: {X_val_raw.shape} | "
                f"Test: {X_test.shape}")

    # ── 3. Preprocess — fit on train ONLY ────────────────────────
    logger.info("\n[3/4] DATA PREPROCESSING")
    """
    # Add target back temporarily so preprocessor can reference it
    train_df = X_train_raw.copy()
    train_df[target] = y_train.values
    val_df   = X_val_raw.copy()
    val_df[target]   = y_val.values
    test_df  = X_test.copy()
    test_df[target]  = y_test.values

    preprocessor = DataPreprocessor(config_path)
    train_clean  = preprocessor.fit_transform(train_df)  # FIT on train only
    val_clean    = preprocessor.transform(val_df)         # APPLY to val
    test_clean   = preprocessor.transform(test_df)        # APPLY to test
    """
    preprocessor = DataPreprocessor(config_path)
    
    train_clean = preprocessor.fit_transform(X_train_raw)
    val_clean = preprocessor.transform(X_val_raw)
    test_clean = preprocessor.transform(X_test)
    preprocessor.save_preprocessor(cfg["paths"]["preprocessor"])
    logger.info("Preprocessor fitted on train only — applied to val and test")

    # ── 4. Feature Engineering — fit on train ONLY ───────────────
    logger.info("\n[4/4] FEATURE ENGINEERING")

    fe          = FeatureEngineer(config_path)
    X_tr_eng    = fe.fit_transform(train_clean)   # FIT on train only
    X_v_eng     = fe.transform(val_clean)        # APPLY to val
    X_te_eng    = fe.transform(test_clean)       # APPLY to test

    #fe.save_scaler(cfg["paths"]["scaler"])
    fe.save_feature_names(cfg["paths"]["feature_names"])
    fe.save_feature_engineer(cfg["paths"]["feature_engineer"])
    fe.save_parquets(X_tr_eng,X_v_eng,X_te_eng,y_train,y_val,y_test)

    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    
    for feature in X_tr_eng.columns:
        print(feature)
    
    
    return {"X_train": X_tr_eng,
        "X_val": X_v_eng,
        "X_test": X_te_eng,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": fe.feature_names_out_,
        "preprocessor": preprocessor,
        "feature_engineer": fe,
        "elapsed_seconds": elapsed,
    }
    
    
if __name__ == "__main__":
    run_pipeline()
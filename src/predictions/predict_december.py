from __future__ import annotations

from pathlib import Path

import json
import joblib
import pandas as pd
from src.data.data_preprocessing import DataPreprocessor
from src.features.build_features  import FeatureEngineer
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


def predict_dec(config_path: str = "dec_config.yaml"):

    cfg = load_config(config_path)
    pp_path  = Path(cfg["paths"]["preprocessor"])
    fe_path = Path(cfg["paths"]["feature_engineer"])
    logger.info("=================  VALIDATION PREDICTIONS =========================")

    # 1. LOAD VALIDATION DATA

    december_path = Path(cfg["paths"]["december_template"])
    december_df = pd.read_csv(december_path)

    logger.info("Validation data loaded: %s",december_df.shape,)
    
    # 2. LOAD BEST MODEL
    
    comparison_path = Path(cfg["artifacts"]["comparison_metrics"])
    with open(comparison_path, "r") as f:
        comparison_results = json.load(f)

    best_model_name = comparison_results["best_model"]

    logger.info( "Best model: %s",best_model_name,)

    model_path = (Path(cfg["artifacts"]["models_dir"])/ f"{best_model_name}{cfg['artifacts']['model_suffix']}")
    model = joblib.load(model_path)

    logger.info("Model loaded from: %s",model_path,)

    
    # 3. APPLY SAME PREPROCESSING / FEATURE ENGINEERING
    
    preprocessor = joblib.load(pp_path)
    X_validation = december_df.drop(columns=["predicted_rate"])
    X_validation = preprocessor.transform(X_validation)
    fe = joblib.load(fe_path)
    X_validation = fe.transform(X_validation)

    # 4. ENSURE SAME FEATURE ORDER

    feature_names_path = Path(cfg["artifacts"]["feature_names"])

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    X_validation = X_validation[feature_names]

    logger.info("Validation features: %s",X_validation.shape,)

    # 5. PREDICT

    predictions = model.predict(X_validation)   

    logger.info("Predictions generated: %d",len(predictions),)

    # 7. MATCH PREDICTION IN THE FILE
    
    if len(predictions) != len(december_df):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) "
            f"does not match December rows ({len(december_df)})"
        )
    
    december_df["predicted_rate"] = predictions
    
    logger.info("Predictions added to 'predicted_rate' column.")
    
    # 8. SAVE FINAL FILE

    output_path = Path(cfg["artifacts"]["december_predictions"])
    output_path.parent.mkdir(parents=True,exist_ok=True,)
    december_df.to_csv(output_path,index=False,)

    logger.info("Predictions saved to: %s",output_path,)
    logger.info("================ FINAL PREDICTION COMPLETED ===============")

    return december_df

if __name__ == "__main__":
    predict_dec()
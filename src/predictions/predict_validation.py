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


def predict_validations(config_path: str = "config.yaml"):

    cfg = load_config(config_path)
    pp_path  = Path(cfg["paths"]["preprocessor"])
    fe_path = Path(cfg["paths"]["feature_engineer"])
    logger.info("=================  VALIDATION PREDICTIONS =========================")

    # 1. LOAD VALIDATION DATA

    validation_path = Path(cfg["paths"]["validation"])
    validation_df = pd.read_csv(validation_path)

    logger.info("Validation data loaded: %s",validation_df.shape,)
    
    # 2. LOAD BEST MODEL
    
    comparison_path = Path(cfg["artifacts"]["comparison_metrics"])
    with open(comparison_path, "r") as f:
        comparison_results = json.load(f)

    best_model_name = comparison_results["best_model"]

    logger.info( "Best model: %s",best_model_name,)

    model_path = (Path(cfg["artifacts"]["models_dir"])/ f"{best_model_name}_regressor.joblib")
    model = joblib.load(model_path)

    logger.info("Model loaded from: %s",model_path,)

    
    # 3. APPLY SAME PREPROCESSING / FEATURE ENGINEERING
    
    preprocessor = joblib.load(pp_path)
    X_validation = preprocessor.transform(validation_df)
    load_ids = X_validation["load_id"].copy()
    X_validation = X_validation.drop(columns=["load_id"])
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
    prediction_df = pd.DataFrame({"load_id": load_ids.values,"predicted_rate": predictions,})

    logger.info("Predictions generated: %d",len(predictions),)

    # 6. LOAD TEMPLATE

    template_path = Path(cfg["paths"]["validation_template"])
    template = pd.read_csv(template_path)

    logger.info("Template loaded: %s",template.shape,)

    # 7. MATCH PREDICTION TO  TEMPLATE BY load_id
    
    prediction_map = dict(zip(prediction_df["load_id"],prediction_df["predicted_rate"],))
    template["predicted_rate"] = template["load_id"].map( prediction_map)

    # 8. SAVE FINAL FILE

    output_path = Path(cfg["artifacts"]["validation_predictions"])
    output_path.parent.mkdir(parents=True,exist_ok=True,)
    template.to_csv(output_path,index=False,)

    logger.info("Predictions saved to: %s",output_path,)
    logger.info("================ FINAL PREDICTION COMPLETED ===============")

    return template

if __name__ == "__main__":
    predict_validations()
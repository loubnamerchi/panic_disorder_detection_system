#loader
from __future__ import annotations

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

import joblib

from src.utils.config import load_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_artifacts() -> dict:
    
    cfg = load_config("config.yaml")
    # ── Experiment 1 artifact paths ───────────────────────────────────
    artifacts_cfg = cfg["artifacts"]
    exp_cfg = artifacts_cfg["experiment_1"]
    
    # 2. LOAD BEST MODEL
    
    comparison_path = Path(exp_cfg["comparison_metrics"])
    with open(comparison_path, "r") as f:
        comparison_results = json.load(f)

    best_model_name = comparison_results["best_model"]

    logger.info( "Best model: %s",best_model_name,)

    model_path = (Path(exp_cfg["models_dir"])/ best_model_name / f"{best_model_name}{artifacts_cfg['model_suffix']}")
    model = joblib.load(model_path)

    logger.info("Model loaded from: %s",model_path,)

    # ── DataPreprocessor ──────────────────────────────────────────────────────
    pp_path  = Path(cfg["paths"]["preprocessor"])
    if not pp_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {pp_path}")
    with open(pp_path, "rb") as f:
        preprocessor = joblib.load(f)
    logger.info("  ✓ preprocessor loaded")

    # ── FeatureEngineer ───────────────────────────────────────────────────────
    fe_path = Path(exp_cfg["feature_engineer"])
    if not fe_path.exists():
        raise FileNotFoundError(f"FeatureEngineer not found: {fe_path}")
    with open(fe_path, "rb") as f:
        feature_engineer = joblib.load(f)
    logger.info("  ✓ feature_engineer loaded")

    # ── Feature names ─────────────────────────────────────────────────────────
    fn_path = Path(exp_cfg["feature_names"])
    feature_names = json.loads(fn_path.read_text()) if fn_path.exists() else []
    logger.info("  ✓ feature_names loaded (%d features)", len(feature_names))
    
    #  ── Metrics ────────────────────────────────────────────────────────
    if best_model_name not in comparison_results:
        raise KeyError(f"Metrics for best model '{best_model_name}' "
                    "were not found in comparison results.")
        
    best_model_results = comparison_results[best_model_name]
    
    if "test" not in best_model_results:
        raise KeyError(f"Test metrics for '{best_model_name}' ""were not found.")
                    
        
    test_results = best_model_results["test"]  
    # Metrics useful for model information / API
    metrics = {
        "pr_auc": test_results["pr_auc"],
        "roc_auc": test_results["roc_auc"],
        "precision": test_results["precision"],
        "recall": test_results["recall"],
        "f1": test_results["f1"],
        "specificity": test_results["specificity"],
        "brier_score": test_results["brier_score"],
        }  
    logger.info(
        "Test metrics | PR-AUC=%.4f | ROC-AUC=%.4f | "
        "Recall=%.4f | F1=%.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["recall"],
        metrics["f1"],
        )                     
                               
    logger.info("All artifacts ready.")
    return {
        "model":            model,
        "preprocessor":     preprocessor,
        "feature_engineer": feature_engineer,
        "feature_names":    feature_names,
        "metrics":          metrics,
    }
    
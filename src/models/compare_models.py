from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.ML.model_factory import create_model
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "brier_score": float(
            brier_score_loss(
                y,
                probabilities,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

def train_and_compare_models(
    models,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    best_params,
    cfg,
):

    random_state = cfg["training"].get(
        "random_state",
        42,
    )

    results = {}

    trained_models = {}

    for model_name in models:

        logger.info(
            "\nTraining final %s model...",
            model_name,
        )

        model = create_model(
            model_name=model_name,
            params=best_params[model_name],
            random_state=random_state,
        )

        # ====================================================
        # FINAL TRAINING
        # ====================================================

        if model_name == "xgboost":

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (X_train, y_train),
                    (X_val, y_val),
                ],
                verbose=False,
            )

        elif model_name == "lightgbm":

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (X_train, y_train),
                    (X_val, y_val),
                ],
                callbacks=[],
            )

        else:

            # Logistic Regression
            # Random Forest

            model.fit(
                X_train,
                y_train,
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_metrics = evaluate_model(
            model,
            X_val,
            y_val,
        )

        # ====================================================
        # TEST
        # ====================================================

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "best_params": best_params[
                model_name
            ],
        }

        trained_models[
            model_name
        ] = model

        # ====================================================
        # LOG RESULTS
        # ====================================================

        logger.info(
            "%s | Val PR-AUC: %.4f | Test PR-AUC: %.4f",
            model_name,
            val_metrics["pr_auc"],
            test_metrics["pr_auc"],
        )

        logger.info(
            "%s | Val ROC-AUC: %.4f | Test ROC-AUC: %.4f",
            model_name,
            val_metrics["roc_auc"],
            test_metrics["roc_auc"],
        )

        logger.info(
            "%s | Val F1: %.4f | Test F1: %.4f",
            model_name,
            val_metrics["f1"],
            test_metrics["f1"],
        )

    # ========================================================
    # SELECT MODEL BASED ON VALIDATION PR-AUC
    # ========================================================

    # PR-AUC is used as the primary comparison metric
    # because the positive class is highly imbalanced.

    best_model_name = max(
        results,
        key=lambda name: results[
            name
        ]["validation"]["pr_auc"],
    )

    results["best_model"] = (
        best_model_name
    )

    # ========================================================
    # SAVE COMPARISON
    # ========================================================

    comparison_path = Path(
        cfg["artifacts"][
            "comparison_metrics"
        ]
    )

    comparison_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # LOG BEST MODEL
    # ========================================================

    logger.info(
        "\n========================================"
    )

    logger.info(
        "BEST MODEL: %s",
        best_model_name,
    )

    logger.info(
        "Validation PR-AUC: %.4f",
        results[
            best_model_name
        ]["validation"]["pr_auc"],
    )

    logger.info(
        "Test PR-AUC: %.4f",
        results[
            best_model_name
        ]["test"]["pr_auc"],
    )

    logger.info(
        "========================================"
    )

    return (
        trained_models,
        results,
    )
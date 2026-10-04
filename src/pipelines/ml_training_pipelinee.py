# ml_training_pipeline

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.models.ML.optuna_tuner import run_optuna
from src.models.evaluate_model import (
    cross_validate_model,
    evaluate_model,
    run_shap_analysis,
)
from src.models.ML.model_factory import (
    create_model,
    get_default_params,
)

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_model(
    config_path: str = "config.yaml",
):

    cfg = load_config(
        config_path
    )

    models = cfg.get(
        "ml_models",
        [
            "logistic_regression",
            "xgboost",
            "lightgbm",
            "random_forest",
        ],
    )

    random_state = cfg[
        "training"
    ].get(
        "random_state",
        42,
    )

    logger.info(
        "================ STARTING ML MODEL TRAINING ====================="
    )

    logger.info(
        "Models: %s",
        models,
    )

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    logger.info(
        "\n[1/6] LOAD PROCESSED DATA"
    )

    # ------------------------------------------------------------
    # Features
    # ------------------------------------------------------------

    # IMPORTANT:
    # All models now receive RAW features.
    #
    # Logistic Regression performs scaling internally through
    # the Pipeline defined in create_model().
    #
    # Therefore, there is no need to load separate
    # X_train_scaled / X_val_scaled / X_test_scaled datasets.

    X_train = pd.read_parquet(
        cfg["paths"]["processed_train"]
    )

    X_val = pd.read_parquet(
        cfg["paths"]["processed_val"]
    )

    X_test = pd.read_parquet(
        cfg["paths"]["processed_test"]
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    y_train = pd.read_parquet(
        cfg["paths"]["processed_y_train"]
    ).squeeze()

    y_val = pd.read_parquet(
        cfg["paths"]["processed_y_val"]
    ).squeeze()

    y_test = pd.read_parquet(
        cfg["paths"]["processed_y_test"]
    ).squeeze()

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    logger.info(
        "Train: %s",
        X_train.shape,
    )

    logger.info(
        "Validation: %s",
        X_val.shape,
    )

    logger.info(
        "Test: %s",
        X_test.shape,
    )

    logger.info(
        "Train positive rate: %.4f",
        y_train.mean(),
    )

    logger.info(
        "Validation positive rate: %.4f",
        y_val.mean(),
    )

    logger.info(
        "Test positive rate: %.4f",
        y_test.mean(),
    )

    feature_names = X_train.columns.tolist()

    logger.info(
        "Number of features: %d",
        len(feature_names),
    )

    # ============================================================
    # 2. OPTUNA HYPERPARAMETER TUNING
    # ============================================================

    logger.info(
        "\n[2/6] OPTUNA HYPERPARAMETER TUNING"
    )

    best_params = {}
    optuna_results = {}

    for model_name in models:

        logger.info(
            "\n----------------------------------------"
        )

        logger.info(
            "OPTUNA: %s",
            model_name.upper(),
        )

        # --------------------------------------------------------
        # ALL MODELS RECEIVE RAW X_train
        # --------------------------------------------------------

        if model_name == "logistic_regression":

            logger.info(
                "%s: StandardScaler is inside model Pipeline",
                model_name,
            )

        else:

            logger.info(
                "%s: using UNSCALED features",
                model_name,
            )

        # --------------------------------------------------------
        # Optuna
        # --------------------------------------------------------

        if cfg["optuna"].get(
            "enabled",
            True,
        ):

            (
                model_best_params,
                best_cv_pr_auc,
                study,
            ) = run_optuna(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                cfg=cfg,
            )

            best_params[
                model_name
            ] = model_best_params

            optuna_results[
                model_name
            ] = {
                "best_cv_pr_auc": float(
                    best_cv_pr_auc
                ),
                "best_params": model_best_params,
                "n_trials": len(
                    study.trials
                ),
            }

            logger.info(
                "%s best CV PR-AUC: %.4f",
                model_name,
                best_cv_pr_auc,
            )

            logger.info(
                "%s best params: %s",
                model_name,
                model_best_params,
            )

        else:

            logger.info(
                "Optuna disabled for %s",
                model_name,
            )

            best_params[
                model_name
            ] = get_default_params(
                model_name,
                cfg,
            )

            optuna_results[
                model_name
            ] = {
                "enabled": False,
                "best_params": best_params[
                    model_name
                ],
            }

    # ============================================================
    # 3. FINAL STRATIFIED CROSS-VALIDATION
    # ============================================================

    logger.info(
        "\n[3/6] FINAL STRATIFIED CROSS-VALIDATION"
    )

    cv_results = {}

    n_splits = cfg[
        "cross_validation"
    ].get(
        "n_splits",
        5,
    )

    for model_name in models:

        logger.info(
            "\nCV: %s",
            model_name.upper(),
        )

        # --------------------------------------------------------
        # CREATE MODEL
        # --------------------------------------------------------

        model = create_model(
            model_name=model_name,
            params=best_params[
                model_name
            ],
            random_state=random_state,
        )

        # --------------------------------------------------------
        # CV ON RAW X_train
        #
        # For Logistic Regression:
        #     Pipeline = StandardScaler + LogisticRegression
        #
        # Therefore scaling occurs independently inside
        # every CV fold.
        # --------------------------------------------------------

        cv_stats = cross_validate_model(
            model=model,
            X=X_train,
            y=y_train,
            n_splits=n_splits,
        )

        cv_results[
            model_name
        ] = cv_stats

        logger.info(
            "%s CV PR-AUC: %.4f ± %.4f",
            model_name,
            cv_stats[
                "cv_pr_auc_mean"
            ],
            cv_stats[
                "cv_pr_auc_std"
            ],
        )

        logger.info(
            "%s CV ROC-AUC: %.4f ± %.4f",
            model_name,
            cv_stats[
                "cv_roc_auc_mean"
            ],
            cv_stats[
                "cv_roc_auc_std"
            ],
        )

        logger.info(
            "%s CV F1: %.4f ± %.4f",
            model_name,
            cv_stats[
                "cv_f1_mean"
            ],
            cv_stats[
                "cv_f1_std"
            ],
        )

        logger.info(
            "%s CV Recall: %.4f ± %.4f",
            model_name,
            cv_stats[
                "cv_recall_mean"
            ],
            cv_stats[
                "cv_recall_std"
            ],
        )

        logger.info(
            "%s CV Precision: %.4f ± %.4f",
            model_name,
            cv_stats[
                "cv_precision_mean"
            ],
            cv_stats[
                "cv_precision_std"
            ],
        )

        logger.info(
            "%s CV Specificity: %.4f ± %.4f",
            model_name,
            cv_stats[
                "cv_specificity_mean"
            ],
            cv_stats[
                "cv_specificity_std"
            ],
        )

    # ============================================================
    # 4. FINAL MODEL TRAINING + COMPARISON
    # ============================================================

    logger.info(
        "\n[4/6] FINAL MODEL TRAINING"
    )

    trained_models = {}
    comparison_results = {}

    for model_name in models:

        logger.info(
            "\nTraining final %s model...",
            model_name,
        )

        # --------------------------------------------------------
        # CREATE MODEL
        # --------------------------------------------------------

        model = create_model(
            model_name=model_name,
            params=best_params[
                model_name
            ],
            random_state=random_state,
        )

        # --------------------------------------------------------
        # FINAL TRAINING
        #
        # ALL MODELS RECEIVE RAW X_train.
        #
        # Logistic Regression:
        #     Pipeline automatically fits scaler on X_train.
        #
        # Tree models:
        #     Train directly on raw features.
        # --------------------------------------------------------

        if model_name == "xgboost":

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (
                        X_train,
                        y_train,
                    ),
                    (
                        X_val,
                        y_val,
                    ),
                ],
                verbose=False,
            )

        elif model_name == "lightgbm":

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (
                        X_train,
                        y_train,
                    ),
                    (
                        X_val,
                        y_val,
                    ),
                ],
                callbacks=[],
            )

        else:

            model.fit(
                X_train,
                y_train,
            )

        trained_models[
            model_name
        ] = model

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        val_metrics = evaluate_model(
            model,
            X_val,
            y_val,
            model_name=model_name,
        )

        # --------------------------------------------------------
        # TEST
        # --------------------------------------------------------

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
            model_name=model_name,
        )

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------

        comparison_results[
            model_name
        ] = {
            "validation": val_metrics,
            "test": test_metrics,
            "best_params": best_params[
                model_name
            ],
        }

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

    # ============================================================
    # 5. ADD CV + OPTUNA RESULTS
    # ============================================================

    logger.info(
        "\n[5/6] BUILD FINAL RESULTS"
    )

    for model_name in models:

        comparison_results[
            model_name
        ]["cv"] = cv_results[
            model_name
        ]

        comparison_results[
            model_name
        ]["optuna"] = optuna_results.get(
            model_name,
            {},
        )

    # ------------------------------------------------------------
    # Select model based on validation PR-AUC
    # ------------------------------------------------------------

    best_model_name = max(
        models,
        key=lambda name: comparison_results[
            name
        ]["validation"]["pr_auc"],
    )

    comparison_results[
        "best_model"
    ] = best_model_name

    # ============================================================
    # SHAP MODEL INTERPRETABILITY
    # ============================================================

    logger.info(
        "\n[SHAP] MODEL INTERPRETABILITY"
    )

    shap_dir = (
        Path(
            cfg["artifacts"]["comparison_metrics"]
        ).parent
        / "shap"
    )

    shap_importance = run_shap_analysis(
        model=trained_models[
            best_model_name
        ],
        X_train=X_train,
        X_test=X_test,
        model_name=best_model_name,
        output_dir=shap_dir,
        random_state=random_state,
    )

    # ============================================================
    # 6. SAVE EVERYTHING
    # ============================================================

    logger.info(
        "\n[6/6] SAVE MODELS AND METRICS"
    )

    models_dir = Path(
        cfg["artifacts"]["models_dir"]
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Save every model
    # ------------------------------------------------------------

    for (
        model_name,
        model,
    ) in trained_models.items():

        model_path = (
            models_dir
            / f"{model_name}"
            f"{cfg['artifacts']['model_suffix']}"
        )

        joblib.dump(
            model,
            model_path,
        )

        logger.info(
            "%s saved to %s",
            model_name,
            model_path,
        )

    # ------------------------------------------------------------
    # Save complete comparison
    # ------------------------------------------------------------

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
            comparison_results,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Save feature names
    # ------------------------------------------------------------

    feature_names_path = Path(
        cfg["artifacts"][
            "feature_names"
        ]
    )

    feature_names_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        feature_names_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_names,
            f,
            indent=2,
        )

    # ============================================================
    # SAVE MODEL COMPARISON TABLE
    # ============================================================

    logger.info(
        "Saving model comparison table..."
    )

    comparison_rows = []

    for model_name in models:

        comparison_rows.append(
            {
                "Model": model_name,

                "Val PR-AUC": comparison_results[
                    model_name
                ]["validation"]["pr_auc"],

                "Val ROC-AUC": comparison_results[
                    model_name
                ]["validation"]["roc_auc"],

                "Val F1": comparison_results[
                    model_name
                ]["validation"]["f1"],

                "Val Recall": comparison_results[
                    model_name
                ]["validation"]["recall"],

                "Test PR-AUC": comparison_results[
                    model_name
                ]["test"]["pr_auc"],

                "Test ROC-AUC": comparison_results[
                    model_name
                ]["test"]["roc_auc"],

                "Test F1": comparison_results[
                    model_name
                ]["test"]["f1"],

                "Test Recall": comparison_results[
                    model_name
                ]["test"]["recall"],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # ------------------------------------------------------------
    # Save table as CSV
    # ------------------------------------------------------------

    comparison_table_path = (
        comparison_path.parent
        / "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_table_path,
        index=False,
    )

    logger.info(
        "Model comparison CSV saved to %s",
        comparison_table_path,
    )

    # ------------------------------------------------------------
    # Save table as PNG figure
    # ------------------------------------------------------------

    comparison_figure_path = (
        comparison_path.parent
        / "15_model_comparison.png"
    )

    fig, ax = plt.subplots(
        figsize=(16, 5)
    )

    ax.axis("off")

    ax.set_title(
        "Final Model Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    table = ax.table(
        cellText=comparison_df.round(4).values,
        colLabels=comparison_df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9
    )

    table.scale(
        1,
        1.8,
    )

    fig.tight_layout()

    fig.savefig(
        comparison_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    logger.info(
        "Model comparison figure saved to %s",
        comparison_figure_path,
    )

    # ============================================================
    # FINAL MODEL COMPARISON
    # ============================================================

    logger.info(
        "================= FINAL MODEL COMPARISON ======================="
    )

    for model_name in models:

        val_pr_auc = comparison_results[
            model_name
        ]["validation"]["pr_auc"]

        test_pr_auc = comparison_results[
            model_name
        ]["test"]["pr_auc"]

        val_roc_auc = comparison_results[
            model_name
        ]["validation"]["roc_auc"]

        test_roc_auc = comparison_results[
            model_name
        ]["test"]["roc_auc"]

        val_f1 = comparison_results[
            model_name
        ]["validation"]["f1"]

        test_f1 = comparison_results[
            model_name
        ]["test"]["f1"]

        val_recall = comparison_results[
            model_name
        ]["validation"]["recall"]

        test_recall = comparison_results[
            model_name
        ]["test"]["recall"]

        logger.info(
            "%s | Val PR-AUC: %.4f | Test PR-AUC: %.4f",
            model_name,
            val_pr_auc,
            test_pr_auc,
        )

        logger.info(
            "%s | Val ROC-AUC: %.4f | Test ROC-AUC: %.4f",
            model_name,
            val_roc_auc,
            test_roc_auc,
        )

        logger.info(
            "%s | Val F1: %.4f | Test F1: %.4f",
            model_name,
            val_f1,
            test_f1,
        )

        logger.info(
            "%s | Val Recall: %.4f | Test Recall: %.4f",
            model_name,
            val_recall,
            test_recall,
        )

    logger.info(
        "BEST MODEL: %s",
        best_model_name,
    )

    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return {
        "models": trained_models,

        "best_model": trained_models[
            best_model_name
        ],

        "best_model_name": best_model_name,

        "best_params": best_params,

        "cv_results": cv_results,

        "optuna_results": optuna_results,

        "comparison_results": comparison_results,

        "X_train": X_train,

        "X_val": X_val,

        "X_test": X_test,

        "y_train": y_train,

        "y_val": y_val,

        "y_test": y_test,

        "feature_names": feature_names,
    }


if __name__ == "__main__":
    train_model()
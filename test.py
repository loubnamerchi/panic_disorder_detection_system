
#model_factory
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_model(
    model_name: str,
    params: dict[str, Any],
    random_state: int = 42,
):

    model_name = model_name.lower()

    if model_name == "xgboost":

        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "lightgbm":

        return LGBMClassifier(
            **params,
            objective="binary",
            metric="average_precision",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    elif model_name == "random_forest":

        return RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "logistic_regression":

        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    **params,
                    random_state=random_state,
                ),
            ),
        ])

    else:

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: xgboost, lightgbm, random_forest, "
            f"logistic_regression."
        )


def get_default_params(
    model_name: str,
    cfg: dict,
) -> dict[str, Any]:

    model_name = model_name.lower()

    if model_name not in cfg["training"]:
        raise ValueError(
            f"No training configuration found for {model_name}"
        )

    return cfg["training"][model_name].copy()



#model_factory
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_model(
    model_name: str,
    params: dict[str, Any],
    random_state: int = 42,
):

    model_name = model_name.lower()

    if model_name == "xgboost":

        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "lightgbm":

        return LGBMClassifier(
            **params,
            objective="binary",
            metric="average_precision",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    elif model_name == "random_forest":

        return RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "logistic_regression":

        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    **params,
                    random_state=random_state,
                ),
            ),
        ])

    else:

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: xgboost, lightgbm, random_forest, "
            f"logistic_regression."
        )


def get_default_params(
    model_name: str,
    cfg: dict,
) -> dict[str, Any]:

    model_name = model_name.lower()

    if model_name not in cfg["training"]:
        raise ValueError(
            f"No training configuration found for {model_name}"
        )

    return cfg["training"][model_name].copy()


#evaluate_model
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.DL.model_factory import (
    MLP,
    FTTransformer,
)


# ============================================================
# DEVICE
# ============================================================

def _get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# MODEL TYPE
# ============================================================

def _is_pytorch_model(model) -> bool:

    return isinstance(
        model,
        torch.nn.Module,
    )


def _is_tabnet_model(model) -> bool:

    return model.__class__.__name__ == "TabNetClassifier"


# ============================================================
# SKLEARN / BOOSTING MODEL
# ============================================================

def _fit_fold_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    if isinstance(model, XGBClassifier):

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False,
        )

    elif isinstance(model, LGBMClassifier):

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
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

    return model


# ============================================================
# PYTORCH PREDICTION
# ============================================================

def _predict_pytorch(
    model,
    X,
    device,
    batch_size=1024,
):

    if hasattr(X, "to_numpy"):

        X = X.to_numpy()

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for (X_batch,) in loader:

            X_batch = X_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    return np.concatenate(
        probabilities
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def _predict_model(
    model,
    X,
):

    # --------------------------------------------------------
    # PyTorch models
    # --------------------------------------------------------

    if _is_pytorch_model(model):

        device = next(
            model.parameters()
        ).device

        probabilities = _predict_pytorch(
            model=model,
            X=X,
            device=device,
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        return predictions, probabilities

    # --------------------------------------------------------
    # TabNet
    # --------------------------------------------------------

    if _is_tabnet_model(model):

        if hasattr(X, "to_numpy"):

            X = X.to_numpy()

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        return predictions, probabilities

    # --------------------------------------------------------
    # Scikit-learn / XGBoost / LightGBM
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    return predictions, probabilities


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def _calculate_classification_metrics(
    y_true,
    predictions,
    probabilities,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    ).ravel()

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return {

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# SKLEARN CROSS-VALIDATION
# ============================================================

def _cross_validate_sklearn_model(
    model,
    X,
    y,
    n_splits,
):

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    fold_metrics = []

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        X_tr = X.iloc[
            train_idx
        ]

        X_val = X.iloc[
            val_idx
        ]

        y_tr = y.iloc[
            train_idx
        ]

        y_val = y.iloc[
            val_idx
        ]

        # ----------------------------------------------------
        # Fresh model
        # ----------------------------------------------------

        fold_model = clone(
            model
        )

        fold_model = _fit_fold_model(
            fold_model,
            X_tr,
            y_tr,
            X_val,
            y_val,
        )

        predictions, probabilities = (
            _predict_model(
                fold_model,
                X_val,
            )
        )

        metrics = (
            _calculate_classification_metrics(
                y_val,
                predictions,
                probabilities,
            )
        )

        fold_metrics.append(
            metrics
        )

    return _aggregate_fold_metrics(
        fold_metrics
    )


# ============================================================
# AGGREGATE CV METRICS
# ============================================================

def _aggregate_fold_metrics(
    fold_metrics,
):

    metric_names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "brier_score",
    ]

    results = {}

    for metric in metric_names:

        scores = [
            fold[metric]
            for fold in fold_metrics
        ]

        results[
            f"cv_{metric}_mean"
        ] = float(
            np.mean(scores)
        )

        results[
            f"cv_{metric}_std"
        ] = float(
            np.std(scores)
        )

    return results


# ============================================================
# EVALUATE FINAL MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str | None = None,
) -> dict[str, Any]:

    predictions, probabilities = (
        _predict_model(
            model,
            X_test,
        )
    )

    metrics = (
        _calculate_classification_metrics(
            y_test,
            predictions,
            probabilities,
        )
    )

    results = {
        "model": model_name,
        **metrics,
    }

    return results


# ============================================================
# CROSS-VALIDATE MODEL
# ============================================================

def cross_validate_model(
    model,
    X,
    y,
    n_splits=5,
):

    # --------------------------------------------------------
    # PyTorch models
    # --------------------------------------------------------

    if _is_pytorch_model(model):

        raise ValueError(
            "PyTorch models should be evaluated "
            "using the trained-model evaluation "
            "pipeline rather than sklearn-style "
            "cross_validate_model()."
        )

    # --------------------------------------------------------
    # TabNet
    # --------------------------------------------------------

    if _is_tabnet_model(model):

        raise ValueError(
            "TabNet should be evaluated using "
            "its dedicated training/evaluation "
            "pipeline because TabNet models "
            "cannot be cloned using sklearn.clone()."
        )

    # --------------------------------------------------------
    # sklearn / XGBoost / LightGBM
    # --------------------------------------------------------

    return _cross_validate_sklearn_model(
        model=model,
        X=X,
        y=y,
        n_splits=n_splits,
    )


#ml_training_pipeline 
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



#evaluate_model
from __future__ import annotations

from typing import Any

import numpy as np
from src.features.build_features  import FeatureEngineer

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
from sklearn.calibration import calibration_curve

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from imblearn.over_sampling import SMOTENC

from sklearn.pipeline import Pipeline
import scipy.sparse as sp
import lightgbm as lgb
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Force un seul thread pour le test

# ============================================================
# DEVICE
# ============================================================

def prepare_features(
    X_train,
    y_train,
    X_val,
    X_test=None,
    use_smotenc=False,
    random_state=42,
):
    """
    Prepare train/validation/test features.

    If use_smotenc=True:
        Ordinal Encoding
        -> SMOTENC on training data only
        -> OneHot Encoding

    If use_smotenc=False:
        Standard FeatureEngineer transformation.

    Returns:
        X_train_final
        y_train_final
        X_val_final
        X_test_final
        feature_engineer
    """

    fe = FeatureEngineer()

    if use_smotenc:

        # ====================================================
        # 1. ORDINAL ENCODING
        # ====================================================
        """
        X_train_ord = fe.encode_ordinal(
            X_train,
            fit=True,
        )

        X_val_ord = fe.encode_ordinal(
            X_val,
            fit=False,
        )

        X_test_ord = None

        if X_test is not None:
            X_test_ord = fe.encode_ordinal(
                X_test,
                fit=False,
            )
        """
        # ====================================================
        # 2. CATEGORICAL INDICES
        # ====================================================

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        # ====================================================
        # 3. SMOTENC ONLY ON TRAINING DATA
        # ====================================================

        smotenc_sampler = SMOTENC(
            categorical_features=cat_cols,
            random_state=random_state,
        )

        X_train_resampled, y_train_resampled = (
            smotenc_sampler.fit_resample(
                X_train,
                y_train,
            )
        )

        # ====================================================
        # 4. ONE-HOT ENCODING
        # ====================================================

        X_train_final = fe.encode_onehot(
            X_train_resampled,
            fit=True,
        )

        X_val_final = fe.encode_onehot(
            X_val,
            fit=False,
        )

        X_test_final = None

        if X_test is not None:
            X_test_final = fe.encode_onehot(
                X_test,
                fit=False,
            )

    else:

        # ====================================================
        # NO SMOTENC
        # ====================================================

        X_train_final = fe.encode_onehot(X_train,fit=True,)
                    
        
        X_val_final = fe.encode_onehot(
                    X_val,
                    fit=False,
                )
        
        X_test_final = None
        
        if X_test is not None:
                    X_test_final = fe.encode_onehot(
                        X_test,
                        fit=False,
                    )
        y_train_resampled = y_train

    # ========================================================
    # FEATURE CONSISTENCY
    # ========================================================

    if list(X_train_final.columns) != list(
        X_val_final.columns
    ):
        raise ValueError(
            "Train and validation feature columns do not match."
        )

    if X_test_final is not None:

        if list(X_train_final.columns) != list(
            X_test_final.columns
        ):
            raise ValueError(
                "Train and test feature columns do not match."
            )

    return (
        X_train_final,
        y_train_resampled,
        X_val_final,
        X_test_final,
        fe,
    )

# ============================================================
# MODEL TYPE
# ============================================================



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
    """
    X_train = np.ascontiguousarray(
        X_train.to_numpy(dtype=np.float64)
    )

    X_val = np.ascontiguousarray(
        X_val.to_numpy(dtype=np.float64)
    )

    y_train = np.ascontiguousarray(
        np.asarray(y_train, dtype=np.int32)
    )

    y_val = np.ascontiguousarray(
        np.asarray(y_val, dtype=np.int32)
    )
    """
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
# MODEL PREDICTIONS
# ============================================================

def _predict_model(
    model,
    X,
):

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
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(
    y_true,
    predictions,
    model_name: str,
    output_dir: str | Path,
    normalize: bool = False,
):
    """
    Plot and save a confusion matrix.

    Parameters
    ----------
    y_true:
        True binary labels.

    predictions:
        Predicted binary labels.

    model_name:
        Name of the model.

    output_dir:
        Directory where the figure will be saved.

    normalize:
        If True, display row-normalized percentages.
        If False, display raw counts.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    # --------------------------------------------------------
    # Normalize if requested
    # --------------------------------------------------------

    if normalize:

        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        cm_display = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sums != 0,
        )

    else:

        cm_display = cm

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        cm_display,
        interpolation="nearest",
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    fig.colorbar(
        im,
        ax=ax,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    classes = [
        "No Panic Disorder",
        "Panic Disorder",
    ]

    ax.set(
        xticks=np.arange(
            len(classes)
        ),
        yticks=np.arange(
            len(classes)
        ),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=(
            f"{model_name.replace('_', ' ').title()} "
            f"Confusion Matrix"
        ),
    )

    # --------------------------------------------------------
    # Annotate cells
    # --------------------------------------------------------

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            if normalize:

                text = (
                    f"{cm_display[i, j] * 100:.1f}%"
                )

            else:

                text = str(
                    cm[i, j]
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
            )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    filename = (
        f"{model_name}_confusion_matrix"
    )

    if normalize:

        filename += "_normalized"

    filename += ".png"

    output_path = (
        output_dir / filename
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

# ============================================================
# SKLEARN CROSS-VALIDATION
# ============================================================

def _cross_validate_sklearn_model(
    model,
    X,
    y,
    n_splits,
    use_smotenc,
    
):
    """
    Cross-validation for sklearn-compatible models.

    Experiment 1:
        Feature engineering
        -> model

    Experiment 2:
        Ordinal encoding
        -> SMOTENC
        -> OneHotEncoder
        -> model

    Important:
        - Feature engineering is fitted independently per fold.
        - SMOTENC is applied ONLY to the training fold.
        - Validation data is never oversampled.
    """

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

        # =====================================================
        # 1. SPLIT FOLD
        # =====================================================

        X_tr = X.iloc[
            train_idx
        ].copy()

        X_val = X.iloc[
            val_idx
        ].copy()

        y_tr = y.iloc[
            train_idx
        ].copy()

        y_val = y.iloc[
            val_idx
        ].copy()
        

        X_tr, y_tr, X_val, _ , fe = prepare_features(X_train=X_tr,
                                    y_train=y_tr,
                                    X_val=X_val,
                                    X_test=None,
                                    use_smotenc=use_smotenc,
                                    random_state=42,
                                )
        X_tr = np.ascontiguousarray(X_tr.values, dtype=np.float32)
        X_val = np.ascontiguousarray(X_val.values, dtype=np.float32)
        y_tr = np.ascontiguousarray(y_tr.values if hasattr(y_tr, 'values') else y_tr, dtype=np.float32).ravel()
        y_val = np.ascontiguousarray(y_val.values if hasattr(y_val, 'values') else y_val, dtype=np.float32).ravel()

        # =====================================================
        # 6. FRESH MODEL
        # =====================================================

        fold_model = clone(
            model
        )

        # =====================================================
        # 7. TRAIN
        # =====================================================

        fold_model = _fit_fold_model(
            fold_model,
            X_tr,
            y_tr,
            X_val,
            y_val,
        )

        # =====================================================
        # 8. PREDICTION
        # =====================================================

        predictions, probabilities = (
            _predict_model(
                fold_model,
                X_val,
            )
        )

        # =====================================================
        # 9. METRICS
        # =====================================================

        metrics = (
            _calculate_classification_metrics(
                y_val,
                predictions,
                probabilities,
            )
        )

        metrics["fold"] = fold

        fold_metrics.append(
            metrics
        )

    # =========================================================
    # 10. AGGREGATE
    # =========================================================

    results = _aggregate_fold_metrics(
        fold_metrics
    )

    return results

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
    confusion_matrix_dir: str | Path | None = None,
    calibration_dir: str | Path | None = None,
    bootstrap_ci: bool = False,
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


    if model_name is not None and confusion_matrix_dir is not None:

        plot_confusion_matrix(
            y_true=y_test,
            predictions=predictions,
            model_name=model_name ,
            output_dir=confusion_matrix_dir,
            normalize=False,
        )

        plot_confusion_matrix(
            y_true=y_test,
            predictions=predictions,
            model_name=model_name ,
            output_dir=confusion_matrix_dir,
            normalize=True,
        )
        
    if model_name is not None and calibration_dir is not None:
        calibration_curve_plot(
            y_true=y_test,
            probabilities=probabilities,
            model_name=model_name,
            output_dir=calibration_dir,
        )
        
    confidence_intervals = None

    if bootstrap_ci:

        confidence_intervals = bootstrap_confidence_intervals(
            y_true=y_test,
            predictions=predictions,
            probabilities=probabilities,
            n_bootstrap=2000,
            confidence_level=0.95,
            random_state=42,
        )
    

    results = {
        "model": model_name,
        **metrics,
    }
    if confidence_intervals is not None:
        results["confidence_intervals"] = confidence_intervals

    return results


# ============================================================
# CROSS-VALIDATE MODEL
# ============================================================

def cross_validate_model(
    model,
    X,
    y,
    use_smotenc,
    n_splits=5,
):

    # --------------------------------------------------------
    # sklearn / XGBoost / LightGBM
    # --------------------------------------------------------

    return _cross_validate_sklearn_model(
        model=model,
        X=X,
        y=y,
        n_splits=n_splits,
        use_smotenc=use_smotenc,
    )

def run_shap_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
    random_state: int = 42,
    background_size: int = 1000,
    explain_size: int = 2000,
):
    """
    Perform SHAP-based model interpretability analysis.

    Parameters
    ----------
    model:
        Fitted model.

    X_train:
        Training features.

    X_test:
        Unseen test features.

    model_name:
        Name of the model:
        xgboost, lightgbm, random_forest,
        or logistic_regression.

    output_dir:
        Directory where SHAP outputs are saved.

    random_state:
        Random seed used for reproducible sampling.

    background_size:
        Number of training observations used as SHAP background.

    explain_size:
        Number of test observations explained.

    Returns
    -------
    pd.DataFrame
        Features ranked by mean absolute SHAP value.
    """

    model_name = model_name.lower()

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. SAMPLE DATA
    # ========================================================

    background = X_train.sample(
        n=min(
            background_size,
            len(X_train),
        ),
        random_state=random_state,
    )

    X_explain = X_test.sample(
        n=min(
            explain_size,
            len(X_test),
        ),
        random_state=random_state,
    )

    # ========================================================
    # 2. CREATE SHAP EXPLAINER
    # ========================================================

    if model_name in {
        "xgboost",
        "lightgbm",
        "random_forest",
    }:

        # ----------------------------------------------------
        # Tree-based models
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer(
            X_explain
        )

    elif model_name == "logistic_regression":

        # ----------------------------------------------------
        # Logistic Regression pipeline:
        #
        # StandardScaler
        #       ↓
        # LogisticRegression
        # ----------------------------------------------------

        if not isinstance(
            model,
            Pipeline,
        ):
            raise TypeError(
                "Logistic Regression model must "
                "be a sklearn Pipeline."
            )

        scaler = model.named_steps[
            "scaler"
        ]

        classifier = model.named_steps[
            "classifier"
        ]

        # Use the already-fitted scaler.
        background_scaled = scaler.transform(
            background
        )

        X_explain_scaled = scaler.transform(
            X_explain
        )

        explainer = shap.LinearExplainer(
            classifier,
            background_scaled,
        )

        shap_values = explainer(
            X_explain_scaled
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # SHAP now sees scaled data, whose feature names
        # are not automatically preserved.
        #
        # Restore the original feature names.
        # ----------------------------------------------------

        X_explain_for_shap = pd.DataFrame(
            X_explain_scaled,
            columns=X_explain.columns,
            index=X_explain.index,
        )

        shap_values = shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=X_explain_for_shap.values,
            feature_names=X_explain.columns.tolist(),
        )

    else:

        raise ValueError(
            f"Unsupported model for SHAP: "
            f"{model_name}"
        )

    # ========================================================
    # 3. SHAP BEESWARM SUMMARY
    # ========================================================

    shap_summary_path = (
        output_dir
        / f"{model_name}_shap_summary.png"
    )

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        shap_summary_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # 4. SHAP GLOBAL IMPORTANCE BAR PLOT
    # ========================================================

    shap_importance_path = (
        output_dir
        / f"{model_name}_shap_importance.png"
    )

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        shap_importance_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # 5. CALCULATE MEAN ABSOLUTE SHAP
    # ========================================================

    mean_abs_shap = (
        pd.DataFrame(
            {
                "Feature": X_explain.columns,
                "Mean_Absolute_SHAP": (
                    abs(shap_values.values)
                    .mean(axis=0)
                ),
            }
        )
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 6. SAVE SHAP IMPORTANCE TABLE
    # ========================================================

    shap_csv_path = (
        output_dir
        / f"{model_name}_shap_importance.csv"
    )

    mean_abs_shap.to_csv(
        shap_csv_path,
        index=False,
    )

    return mean_abs_shap

def calibration_curve_plot(
    y_true,
    probabilities,
    model_name: str,
    output_dir: str | Path,
    n_bins: int = 10,
):
    """
    Plot and save a reliability/calibration curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    probabilities : array-like
        Predicted probabilities for the positive class.

    model_name : str
        Name of the model.

    output_dir : str or Path
        Directory where the plot will be saved.

    n_bins : int
        Number of probability bins.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration reference
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfectly calibrated",
    )

    # Model calibration
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label=model_name.replace("_", " ").title(),
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} Calibration Curve"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    output_path = (
        output_dir
        / f"{model_name}_calibration_curve.png"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path

def bootstrap_confidence_intervals(
    y_true,
    predictions,
    probabilities,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Estimate bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    y_true:
        True binary labels.

    predictions:
        Binary predictions.

    probabilities:
        Predicted probability for the positive class.

    n_bootstrap:
        Number of bootstrap resamples.

    confidence_level:
        Confidence level for the interval.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Bootstrap mean, lower CI, and upper CI for each metric.
    """

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(random_state)

    n_samples = len(y_true)

    bootstrap_metrics = {}

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

    for metric_name in metric_names:
        bootstrap_metrics[metric_name] = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n_samples,
            size=n_samples,
        )

        y_boot = y_true[indices]
        predictions_boot = predictions[indices]
        probabilities_boot = probabilities[indices]

        # ROC-AUC and PR-AUC require both classes.
        if len(np.unique(y_boot)) < 2:
            continue

        metrics = _calculate_classification_metrics(
            y_boot,
            predictions_boot,
            probabilities_boot,
        )

        for metric_name in metric_names:
            bootstrap_metrics[metric_name].append(
                metrics[metric_name]
            )

    alpha = 1.0 - confidence_level

    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    confidence_intervals = {}

    for metric_name, values in bootstrap_metrics.items():

        values = np.asarray(values)

        confidence_intervals[metric_name] = {
        
            "lower": float(
                np.percentile(
                    values,
                    lower_percentile,
                )
            ),
            "upper": float(
                np.percentile(
                    values,
                    upper_percentile,
                )
            ),
        }

    return confidence_intervals
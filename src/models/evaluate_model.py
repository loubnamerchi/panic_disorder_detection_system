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

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.pipeline import Pipeline


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
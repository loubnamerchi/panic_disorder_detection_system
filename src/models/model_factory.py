from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

        return LogisticRegression(
            **params,
            random_state=random_state,
        )

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
from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from optuna.integration import (
    XGBoostPruningCallback,
    LightGBMPruningCallback,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from src.models.model_factory import create_model


def run_optuna(
    model_name: str,
    X_train,
    y_train,
    cfg: dict,
):

    model_name = model_name.lower()

    n_splits = cfg["cross_validation"].get(
        "n_splits",
        5,
    )

    n_trials = cfg["optuna"].get(
        "n_trials",
        50,
    )

    random_state = cfg["training"].get(
        "random_state",
        42,
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    optuna_cfg = cfg["optuna"][model_name]

    def objective(trial):

        # XGBOOST

        if model_name == "xgboost":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_cfg["n_estimators_min"],
                    optuna_cfg["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_cfg["max_depth_min"],
                    optuna_cfg["max_depth_max"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_cfg["learning_rate_min"],
                    optuna_cfg["learning_rate_max"],
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    optuna_cfg["subsample_min"],
                    optuna_cfg["subsample_max"],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    optuna_cfg["colsample_bytree_min"],
                    optuna_cfg["colsample_bytree_max"],
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    optuna_cfg["min_child_weight_min"],
                    optuna_cfg["min_child_weight_max"],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    optuna_cfg["reg_alpha_min"],
                    optuna_cfg["reg_alpha_max"],
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    optuna_cfg["reg_lambda_min"],
                    optuna_cfg["reg_lambda_max"],
                    log=True,
                ),
            }

        # LIGHTGBM

        elif model_name == "lightgbm":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_cfg["n_estimators_min"],
                    optuna_cfg["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_cfg["max_depth_min"],
                    optuna_cfg["max_depth_max"],
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    optuna_cfg["num_leaves_min"],
                    optuna_cfg["num_leaves_max"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_cfg["learning_rate_min"],
                    optuna_cfg["learning_rate_max"],
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    optuna_cfg["subsample_min"],
                    optuna_cfg["subsample_max"],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    optuna_cfg["colsample_bytree_min"],
                    optuna_cfg["colsample_bytree_max"],
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    optuna_cfg["min_child_samples_min"],
                    optuna_cfg["min_child_samples_max"],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    optuna_cfg["reg_alpha_min"],
                    optuna_cfg["reg_alpha_max"],
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    optuna_cfg["reg_lambda_min"],
                    optuna_cfg["reg_lambda_max"],
                    log=True,
                ),
            }

        # RANDOM FOREST

        elif model_name == "random_forest":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_cfg["n_estimators_min"],
                    optuna_cfg["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    optuna_cfg["max_depth_min"],
                    optuna_cfg["max_depth_max"],
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    optuna_cfg["min_samples_split_min"],
                    optuna_cfg["min_samples_split_max"],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    optuna_cfg["min_samples_leaf_min"],
                    optuna_cfg["min_samples_leaf_max"],
                ),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    optuna_cfg["max_features_options"],
                ),
            }

        else:

            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        # STRATIFIED CV

        fold_pr_auc = []

        for fold, (
            train_idx,
            valid_idx,
        ) in enumerate(
            skf.split(X_train, y_train),
            start=1,
        ):

            X_tr = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[valid_idx]

            y_tr = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[valid_idx]

            model = create_model(
                model_name=model_name,
                params=params,
                random_state=random_state,
            )

            # XGBoost

            if model_name == "xgboost":

                pruning_callback = (
                    XGBoostPruningCallback(
                        trial,
                        "validation_0-aucpr",
                    )
                )

                model.set_params(
                    callbacks=[pruning_callback]
                )

                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[
                        (X_fold_val, y_fold_val)
                    ],
                    verbose=False,
                )

            # LightGBM

            elif model_name == "lightgbm":

                pruning_callback = (
                    LightGBMPruningCallback(
                        trial,
                        "average_precision",
                    )
                )

                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[
                        (X_fold_val, y_fold_val)
                    ],
                    callbacks=[
                        pruning_callback
                    ],
                )

            # Random Forest

            else:

                model.fit(
                    X_tr,
                    y_tr,
                )

            probabilities = model.predict_proba(
                X_fold_val
            )[:, 1]

            pr_auc = average_precision_score(
                y_fold_val,
                probabilities,
            )

            fold_pr_auc.append(pr_auc)

            intermediate_pr_auc = np.mean(
                fold_pr_auc
            )

            if model_name in [
                "xgboost",
                "lightgbm",
            ]:

                trial.report(
                    intermediate_pr_auc,
                    step=fold,
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

        return float(
            np.mean(fold_pr_auc)
        )

    # OPTUNA STUDY

    if model_name in [
        "xgboost",
        "lightgbm",
    ]:

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_classification",
            sampler=optuna.samplers.TPESampler(
                seed=random_state
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
            ),
        )

    else:

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_classification",
            sampler=optuna.samplers.TPESampler(
                seed=random_state
            ),
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return (
        study.best_params,
        study.best_value,
        study,
    )
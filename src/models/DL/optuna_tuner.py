from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models.DL.model_factory import create_model


# ============================================================
# Utility functions
# ============================================================

def _get_device() -> torch.device:
    """
    Select the available computation device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _to_numpy(data) -> np.ndarray:
    """
    Convert pandas/numpy-like data to float32 numpy array.
    """

    if hasattr(data, "to_numpy"):
        data = data.to_numpy()

    return np.asarray(data, dtype=np.float32)


def _to_numpy_target(data) -> np.ndarray:
    """
    Convert target to 1D numpy array.
    """

    if hasattr(data, "to_numpy"):
        data = data.to_numpy()

    return np.asarray(data).reshape(-1)


# ============================================================
# PyTorch prediction
# ============================================================

def _predict_torch(
    model: nn.Module,
    X,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Generate positive-class probabilities for a PyTorch model.
    """

    X_np = _to_numpy(X)

    dataset = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32)
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

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            probabilities.append(
                probs.cpu().numpy()
            )

    return np.concatenate(probabilities)


# ============================================================
# PyTorch training
# ============================================================

def _train_torch_model(
    model: nn.Module,
    X_train,
    y_train,
    X_valid,
    y_valid,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
    trial: optuna.Trial | None = None,
    trial_step_offset: int = 0,
):
    """
    Train a PyTorch binary-classification model.

    Early stopping is based on validation PR-AUC.

    Returns
    -------
    model:
        Model restored to the best validation epoch.
    best_pr_auc:
        Best validation PR-AUC achieved during training.
    """

    X_train_np = _to_numpy(X_train)
    y_train_np = _to_numpy_target(y_train).astype(
        np.float32
    )

    X_valid_np = _to_numpy(X_valid)
    y_valid_np = _to_numpy_target(y_valid).astype(
        np.float32
    )

    train_dataset = TensorDataset(
        torch.tensor(
            X_train_np,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train_np,
            dtype=torch.float32,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_pr_auc = -np.inf
    best_state = None

    epochs_without_improvement = 0

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_probabilities = _predict_torch(
            model=model,
            X=X_valid,
            device=device,
            batch_size=batch_size,
        )

        validation_pr_auc = average_precision_score(
            y_valid_np,
            validation_probabilities,
        )

        # ----------------------------------------------------
        # Optuna pruning
        # ----------------------------------------------------

        if trial is not None:

            trial.report(
                validation_pr_auc,
                step=trial_step_offset + epoch,
            )

            if trial.should_prune():

                raise optuna.TrialPruned()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if validation_pr_auc > best_pr_auc:

            best_pr_auc = validation_pr_auc

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, float(best_pr_auc)


# ============================================================
# MLP Optuna parameters
# ============================================================

def _suggest_mlp_params(
    trial: optuna.Trial,
    optuna_cfg: dict[str, Any],
) -> dict[str, Any]:

    n_layers = trial.suggest_int(
        "n_layers",
        optuna_cfg["n_layers_min"],
        optuna_cfg["n_layers_max"],
    )

    hidden_units = trial.suggest_int(
        "hidden_units",
        optuna_cfg["hidden_units_min"],
        optuna_cfg["hidden_units_max"],
    )

    hidden_layers = [
        hidden_units
        for _ in range(n_layers)
    ]

    return {
        "hidden_layers": hidden_layers,

        "dropout": trial.suggest_float(
            "dropout",
            optuna_cfg["dropout_min"],
            optuna_cfg["dropout_max"],
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            optuna_cfg["learning_rate_min"],
            optuna_cfg["learning_rate_max"],
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            optuna_cfg["weight_decay_min"],
            optuna_cfg["weight_decay_max"],
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            optuna_cfg["batch_size_options"],
        ),

        "activation": trial.suggest_categorical(
            "activation",
            optuna_cfg["activation_options"],
        ),
    }


# ============================================================
# FT-Transformer Optuna parameters
# ============================================================

def _suggest_ft_transformer_params(
    trial: optuna.Trial,
    optuna_cfg: dict[str, Any],
) -> dict[str, Any]:

    return {
        "d_token": trial.suggest_categorical(
            "d_token",
            optuna_cfg["d_token_options"],
        ),

        "n_blocks": trial.suggest_int(
            "n_blocks",
            optuna_cfg["n_blocks_min"],
            optuna_cfg["n_blocks_max"],
        ),

        "attention_n_heads": trial.suggest_categorical(
            "attention_n_heads",
            optuna_cfg["attention_n_heads_options"],
        ),

        "attention_dropout": trial.suggest_float(
            "attention_dropout",
            optuna_cfg["attention_dropout_min"],
            optuna_cfg["attention_dropout_max"],
        ),

        "ffn_dropout": trial.suggest_float(
            "ffn_dropout",
            optuna_cfg["ffn_dropout_min"],
            optuna_cfg["ffn_dropout_max"],
        ),

        "ffn_factor": trial.suggest_int(
            "ffn_factor",
            optuna_cfg["ffn_factor_min"],
            optuna_cfg["ffn_factor_max"],
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            optuna_cfg["learning_rate_min"],
            optuna_cfg["learning_rate_max"],
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            optuna_cfg["weight_decay_min"],
            optuna_cfg["weight_decay_max"],
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            optuna_cfg["batch_size_options"],
        ),
    }


# ============================================================
# TabNet Optuna parameters
# ============================================================

def _suggest_tabnet_params(
    trial: optuna.Trial,
    optuna_cfg: dict[str, Any],
) -> dict[str, Any]:

    return {
        "n_d": trial.suggest_int(
            "n_d",
            optuna_cfg["n_d_min"],
            optuna_cfg["n_d_max"],
        ),

        "n_a": trial.suggest_int(
            "n_a",
            optuna_cfg["n_a_min"],
            optuna_cfg["n_a_max"],
        ),

        "n_steps": trial.suggest_int(
            "n_steps",
            optuna_cfg["n_steps_min"],
            optuna_cfg["n_steps_max"],
        ),

        "gamma": trial.suggest_float(
            "gamma",
            optuna_cfg["gamma_min"],
            optuna_cfg["gamma_max"],
        ),

        "n_independent": trial.suggest_int(
            "n_independent",
            optuna_cfg["n_independent_min"],
            optuna_cfg["n_independent_max"],
        ),

        "n_shared": trial.suggest_int(
            "n_shared",
            optuna_cfg["n_shared_min"],
            optuna_cfg["n_shared_max"],
        ),

        "lambda_sparse": trial.suggest_float(
            "lambda_sparse",
            optuna_cfg["lambda_sparse_min"],
            optuna_cfg["lambda_sparse_max"],
            log=True,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            optuna_cfg["learning_rate_min"],
            optuna_cfg["learning_rate_max"],
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            optuna_cfg["batch_size_options"],
        ),

        "virtual_batch_size": trial.suggest_categorical(
            "virtual_batch_size",
            optuna_cfg[
                "virtual_batch_size_options"
            ],
        ),
    }


# ============================================================
# Main Optuna function
# ============================================================

def run_optuna(
    model_name: str,
    X_train,
    y_train,
    cfg: dict,
):
    """
    Run Optuna hyperparameter optimization for a
    deep-learning model.

    Supported models:
        - mlp
        - tabnet
        - ft_transformer

    Optimization objective:
        Mean validation PR-AUC across StratifiedKFold.
    """

    model_name = model_name.lower()

    supported_models = [
        "mlp",
        "tabnet",
        "ft_transformer",
    ]

    if model_name not in supported_models:

        raise ValueError(
            f"Unsupported deep-learning model: "
            f"{model_name}. "
            f"Supported models: "
            f"{', '.join(supported_models)}"
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    n_splits = cfg[
        "cross_validation"
    ].get(
        "n_splits",
        5,
    )

    n_trials = cfg[
        "optuna"
    ].get(
        "n_trials",
        50,
    )

    random_state = cfg[
        "training"
    ].get(
        "random_state",
        42,
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    optuna_cfg = cfg[
        "optuna"
    ][
        model_name
    ]

    device = _get_device()

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------------------------
    # Optuna objective
    # --------------------------------------------------------

    def objective(
        trial: optuna.Trial,
    ):

        # ====================================================
        # Hyperparameter search
        # ====================================================

        if model_name == "mlp":

            params = _suggest_mlp_params(
                trial,
                optuna_cfg,
            )

        elif model_name == "ft_transformer":

            params = _suggest_ft_transformer_params(
                trial,
                optuna_cfg,
            )

        elif model_name == "tabnet":

            params = _suggest_tabnet_params(
                trial,
                optuna_cfg,
            )

        else:

            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        # ====================================================
        # Cross-validation
        # ====================================================

        fold_pr_auc = []

        for fold, (
            train_idx,
            valid_idx,
        ) in enumerate(
            skf.split(
                X_train,
                y_train,
            ),
            start=1,
        ):

            # ------------------------------------------------
            # Fold data
            # ------------------------------------------------

            X_tr = X_train.iloc[
                train_idx
            ]

            X_fold_val = X_train.iloc[
                valid_idx
            ]

            y_tr = y_train.iloc[
                train_idx
            ]

            y_fold_val = y_train.iloc[
                valid_idx
            ]

            # =================================================
            # MLP
            # =================================================

            if model_name == "mlp":

                input_dim = X_tr.shape[1]

                model = create_model(
                    model_name="mlp",
                    params=params,
                    input_dim=input_dim,
                    random_state=random_state,
                )

                training_cfg = cfg[
                    "training"
                ][
                    "mlp"
                ]

                model, best_pr_auc = (
                    _train_torch_model(
                        model=model,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_valid=X_fold_val,
                        y_valid=y_fold_val,
                        learning_rate=params[
                            "learning_rate"
                        ],
                        weight_decay=params[
                            "weight_decay"
                        ],
                        batch_size=params[
                            "batch_size"
                        ],
                        epochs=training_cfg.get(
                            "epochs",
                            100,
                        ),
                        patience=training_cfg.get(
                            "patience",
                            15,
                        ),
                        device=device,
                        trial=trial,
                        trial_step_offset=(
                            fold - 1
                        )
                        * training_cfg.get(
                            "epochs",
                            100,
                        ),
                    )
                )

                pr_auc = best_pr_auc

            # =================================================
            # FT-TRANSFORMER
            # =================================================

            elif model_name == "ft_transformer":

                input_dim = X_tr.shape[1]

                model = create_model(
                    model_name="ft_transformer",
                    params=params,
                    input_dim=input_dim,
                    random_state=random_state,
                )

                training_cfg = cfg[
                    "training"
                ][
                    "ft_transformer"
                ]

                model, best_pr_auc = (
                    _train_torch_model(
                        model=model,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_valid=X_fold_val,
                        y_valid=y_fold_val,
                        learning_rate=params[
                            "learning_rate"
                        ],
                        weight_decay=params[
                            "weight_decay"
                        ],
                        batch_size=params[
                            "batch_size"
                        ],
                        epochs=training_cfg.get(
                            "epochs",
                            100,
                        ),
                        patience=training_cfg.get(
                            "patience",
                            15,
                        ),
                        device=device,
                        trial=trial,
                        trial_step_offset=(
                            fold - 1
                        )
                        * training_cfg.get(
                            "epochs",
                            100,
                        ),
                    )
                )

                pr_auc = best_pr_auc

            # =================================================
            # TABNET
            # =================================================

            elif model_name == "tabnet":

                model = create_model(
                    model_name="tabnet",
                    params=params,
                    input_dim=X_tr.shape[1],
                    random_state=random_state,
                )

                training_cfg = cfg[
                    "training"
                ][
                    "tabnet"
                ]

                X_tr_np = _to_numpy(
                    X_tr
                )

                X_fold_val_np = _to_numpy(
                    X_fold_val
                )

                y_tr_np = _to_numpy_target(
                    y_tr
                )

                y_fold_val_np = _to_numpy_target(
                    y_fold_val
                )

                model.fit(
                    X_tr_np,
                    y_tr_np,
                    eval_set=[
                        (
                            X_fold_val_np,
                            y_fold_val_np,
                        )
                    ],
                    eval_name=[
                        "validation"
                    ],
                    eval_metric=[
                        "auc"
                    ],
                    max_epochs=training_cfg.get(
                        "max_epochs",
                        100,
                    ),
                    patience=training_cfg.get(
                        "patience",
                        15,
                    ),
                    batch_size=params[
                        "batch_size"
                    ],
                    virtual_batch_size=params[
                        "virtual_batch_size"
                    ],
                    drop_last=False,
                )

                probabilities = (
                    model.predict_proba(
                        X_fold_val_np
                    )[:, 1]
                )

                pr_auc = average_precision_score(
                    y_fold_val_np,
                    probabilities,
                )

            # ------------------------------------------------
            # Store fold score
            # ------------------------------------------------

            fold_pr_auc.append(
                pr_auc
            )

            intermediate_pr_auc = float(
                np.mean(
                    fold_pr_auc
                )
            )

            # ------------------------------------------------
            # Report fold-level progress
            # ------------------------------------------------

            trial.report(
                intermediate_pr_auc,
                step=fold,
            )

            if trial.should_prune():

                raise optuna.TrialPruned()

        # ====================================================
        # Final objective
        # ====================================================

        return float(
            np.mean(
                fold_pr_auc
            )
        )

    # ========================================================
    # Optuna study
    # ========================================================

    study = optuna.create_study(
        direction="maximize",
        study_name=(
            f"{model_name}_classification"
        ),
        sampler=optuna.samplers.TPESampler(
            seed=random_state
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )

    # ========================================================
    # Run optimization
    # ========================================================

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ========================================================
    # Return results
    # ========================================================

    return (
        study.best_params,
        study.best_value,
        study,
    )
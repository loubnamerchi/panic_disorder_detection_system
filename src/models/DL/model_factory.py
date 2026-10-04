from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim

        activation_layer = self._get_activation(activation)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:

        activation = activation.lower()

        if activation == "relu":
            return nn.ReLU()

        elif activation == "gelu":
            return nn.GELU()

        else:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Supported activations: relu, gelu."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class FTTransformer(nn.Module):
    """
    FT-Transformer for tabular binary classification.

    This implementation uses numerical features as tokens.
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        n_blocks: int = 4,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.2,
        ffn_factor: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_token = d_token

        # Each numerical feature becomes a token.
        self.feature_tokenizer = nn.Linear(1, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=d_token * ffn_factor,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_blocks,
        )

        self.dropout = nn.Dropout(ffn_dropout)

        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # [batch, features]
        # → [batch, features, 1]
        x = x.unsqueeze(-1)

        # [batch, features, 1]
        # → [batch, features, d_token]
        x = self.feature_tokenizer(x)

        # Transformer
        x = self.transformer(x)

        # Mean pooling over feature tokens
        x = x.mean(dim=1)

        x = self.dropout(x)

        return self.head(x).squeeze(1)


def create_model(
    model_name: str,
    params: dict[str, Any],
    input_dim: int,
    random_state: int = 42,
):
    """
    Create a deep-learning model.

    Parameters
    ----------
    model_name:
        One of: mlp, tabnet, ft_transformer.

    params:
        Model hyperparameters.

    input_dim:
        Number of input features.

    random_state:
        Random seed.
    """

    model_name = model_name.lower()

    if model_name == "mlp":

        return MLP(
            input_dim=input_dim,
            hidden_layers=params.get(
                "hidden_layers",
                [128, 64],
            ),
            dropout=params.get(
                "dropout",
                0.2,
            ),
            activation=params.get(
                "activation",
                "relu",
            ),
        )

    elif model_name == "tabnet":

        return TabNetClassifier(
            n_d=params.get("n_d", 32),
            n_a=params.get("n_a", 32),
            n_steps=params.get("n_steps", 5),
            gamma=params.get("gamma", 1.5),
            n_independent=params.get("n_independent", 2),
            n_shared=params.get("n_shared", 2),
            lambda_sparse=params.get(
                "lambda_sparse",
                0.0001,
            ),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={
                "lr": params.get(
                    "learning_rate",
                    0.02,
                )
            },
            mask_type="entmax",
            seed=random_state,
            verbose=0,
        )

    elif model_name == "ft_transformer":

        return FTTransformer(
            input_dim=input_dim,
            d_token=params.get(
                "d_token",
                64,
            ),
            n_blocks=params.get(
                "n_blocks",
                4,
            ),
            attention_n_heads=params.get(
                "attention_n_heads",
                8,
            ),
            attention_dropout=params.get(
                "attention_dropout",
                0.2,
            ),
            ffn_dropout=params.get(
                "ffn_dropout",
                0.2,
            ),
            ffn_factor=params.get(
                "ffn_factor",
                4,
            ),
        )

    else:

        raise ValueError(
            f"Unknown deep-learning model: {model_name}. "
            f"Supported models: mlp, tabnet, ft_transformer."
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
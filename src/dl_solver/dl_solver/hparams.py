from typing import List, Tuple, Union

import optuna
from pydantic import BaseModel, Field, model_validator


class OptimizableHParam(BaseModel):
    """Meta class for hyperparameters that can be optimized with Optuna."""

    dtype: Union[type, str]
    upper: float = 1.0
    lower: float = 0.0
    optim_mode: str = "ignore"
    candidate_values: List[Union[int, float, bool]] = []

    def suggest(self, trial: optuna.trial.Trial, name: str):
        if self.optim_mode == "ignore":
            return self.lower

        if self.dtype == "int":
            return trial.suggest_int(name, int(self.lower), int(self.upper))

        if self.dtype == "float":
            return trial.suggest_float(name, self.lower, self.upper)

        if self.dtype == "bool":
            return trial.suggest_categorical(name, [True, False])

        if self.dtype == "categorical":
            return trial.suggest_categorical(name, self.candidate_values)

        raise ValueError(f"Unknown dtype: {self.dtype}")


class HyperParameters(BaseModel):
    """Container for all hyperparameters."""

    class Transformer(BaseModel):
        d_model: int = 1024  # Increased from 768 to 1024
        nhead: int = 8
        num_encoder_layers: int = 8  # Increased from 6 to 8
        num_decoder_layers: int = 8  # Increased from 6 to 8
        dim_feedforward: int = 2048  # Increased from 2048 to 3072

        @model_validator(mode="after")
        def __validate(self) -> "HyperParameters.Transformer":
            assert self.d_model % self.nhead == 0
            assert self.dim_feedforward % self.nhead == 0

            return self

    class Backbone(BaseModel):
        num_features_out: int = Field(
            default_factory=lambda: HyperParameters.Transformer().d_model
        )
        is_trainable: bool = False

    class IdxClassifier(BaseModel):
        input_features: int = Field(
            default_factory=lambda: HyperParameters.Transformer().d_model
        )
        max_rows: int = 3
        max_cols: int = 4
        f_dim: int = 256  # Increased from 128 to 256
        h_dim: int = 512  # Increased from 256 to 512
        g_dim: int = 1

    class FourierEmbedding(BaseModel):
        pos_dim: int = 1
        f_dim: int = 128  # Increased from 64 to 128
        h_dim: int = 256  # Increased from 128 to 256
        d_dim: int = Field(
            default_factory=lambda: HyperParameters.Transformer().d_model
        )
        g_dim: int = 1
        gamma: float = 1

        @model_validator(mode="after")
        def __validate(self) -> "HyperParameters.FourierEmbedding":
            assert (
                self.f_dim % 2 == 0
            ), "number of fourier feature dimensions must be divisible by 2."
            assert (
                self.d_dim % self.g_dim == 0
            ), "number of D dimension must be divisible by the number of G dimension."
            return self

    class TypeClassifier(BaseModel):
        fourier_embedding: "HyperParameters.FourierEmbedding" = Field(
            default_factory=lambda: HyperParameters.FourierEmbedding(
                pos_dim=3, f_dim=12, d_dim=768
            )
        )
        input_features: int = Field(
            default_factory=lambda: HyperParameters.Transformer().d_model
        )

    class Optimizer(BaseModel):
        batch_size: int = 132

        # Learning Rates
        lr_backbone: float = 5e-5
        lr_transformer: float = 1e-4
        lr_classifier: float = 1e-4

        weight_decay: float = 1e-4

    class Criteria(BaseModel):
        is_norm_costs: bool = True
        w_mse_loss: float = 4
        w_ce_rot_loss: float = 1
        w_ce_pos_loss: float = 1
        w_ce_type_loss: float = 1
        w_unique_loss: float = 1.5
        unique_cost_sigma: float = 0.5

    backbone: Backbone = Backbone()
    transformer: Transformer = Transformer()
    classifier: IdxClassifier = IdxClassifier()
    type_classifier: TypeClassifier = TypeClassifier()
    fourier_embedding_pos: FourierEmbedding = FourierEmbedding(pos_dim=2)
    fourier_embedding_rot: FourierEmbedding = FourierEmbedding(
        pos_dim=1, h_dim=2, f_dim=2
    )
    optimizer: Optimizer = Optimizer()
    criteria: Criteria = Criteria()
    puzzle_shape: Tuple[int, int] = Field(default=(3, 4), __doc__="Rows x Cols")
    segment_shape: Tuple[int, int] = (48, 48)

    @model_validator(mode="after")
    def __validate(self) -> "HyperParameters":
        assert self.type_classifier.fourier_embedding.d_dim == self.transformer.d_model
        assert self.fourier_embedding_pos.d_dim == self.transformer.d_model
        assert self.fourier_embedding_rot.d_dim == self.transformer.d_model
        assert self.classifier.input_features == self.transformer.d_model
        return self

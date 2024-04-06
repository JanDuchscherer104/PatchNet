from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Type, TypeVar

import mlflow
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from typing_extensions import Annotated

T = TypeVar("T", bound="YamlBaseModel")


class YamlBaseModel(BaseModel):
    @classmethod
    def from_yaml(cls: Type[T], file: Path | str) -> T:
        return cls.model_validate(parse_yaml_file_as(cls, file))

    def to_yaml(self, file: Path | str) -> None:
        to_yaml_file(file, self, indent=4)


class Paths(YamlBaseModel):
    root: Path = Field(default_factory=lambda: Path(__file__).parents[5].resolve())
    data: Annotated[Path, Field(default=".data/imagenet", validate_default=True)]
    checkpoints: Annotated[
        Path, Field(default="src/solver/.logs/checkpoints", validate_default=True)
    ]
    tb_logs: Annotated[
        Path, Field(default="src/solver/.logs/tb_logs", validate_default=True)
    ]
    mlflow_uri: Annotated[
        str, Field(default="src/solver/.logs/mlflow_logs/mlflow", validate_default=True)
    ]

    @field_validator("data", "checkpoints", "tb_logs")
    @classmethod
    def __convert_to_path(cls, v: str, info: ValidationInfo) -> Path:
        root = info.data.get("root")
        v = root / v if not Path(v).is_absolute() else Path(v)
        if v == "data":
            assert v.exists(), f"Data directory {v} does not exist."
        else:
            v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @field_validator("mlflow_uri")
    @classmethod
    def __convert_to_uri(cls, v: str, info: ValidationInfo) -> str:
        if v.startswith("file://"):
            return v
        root = info.data.get("root")
        v: Path = root / v if not Path(v).is_absolute() else Path(v)
        v.parent.mkdir(parents=True, exist_ok=True)
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return (
            v.resolve().as_uri()
        )  # .replace("file:///", "sqlite:///") # TODO: Fix this


class MLflowConfig(YamlBaseModel):
    experiment_name: str = "DL-EXP"
    run_name: str = Annotated[str, Field(default=None)]
    experiment_id: Annotated[str, Field(default=None)]


class Config(YamlBaseModel):
    is_debug: bool = False
    verbose: bool = True
    from_ckpt: Optional[str] = None
    is_multiproc: bool = True
    is_optuna: bool = True
    num_workers: Optional[int] = 4
    pin_memory: bool = True
    max_epochs: int = 50
    is_gpu: bool = True
    log_every_n_steps: int = 128
    is_fast_dev_run: bool = False
    active_callbacks: Dict[
        Literal[
            "ModelCheckpoint",
            "TQDMProgressBar",
            "EarlyStopping",
            "BatchSizeFinder",
            "OptunaPruning",
            "LearningRateMonitor",
            "ModelSummary",
        ],
        bool,
    ] = {
        "ModelCheckpoint": False,
        "TQDMProgressBar": False,
        "EarlyStopping": True,
        "BatchSizeFinder": False,
        "LearningRateMonitor": True,
        "ModelSummary": True,
        "OptunaPruning": False,
    }
    paths: Paths = Field(default_factory=Paths)
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)

    @model_validator(mode="after")
    def __setup_mlflow(self) -> "Config":
        mlflow.set_tracking_uri(self.paths.mlflow_uri)
        experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(self.mlflow_config.experiment_name)
        )
        self.mlflow_config.experiment_id = experiment_id
        last_run = mlflow.search_runs(
            order_by=["start_time DESC"], max_results=1, experiment_ids=[experiment_id]
        )

        if last_run.empty:
            next_run_num = 1
        else:
            last_run_label = last_run.iloc[0]["tags.mlflow.runName"]
            last_run_num = int(last_run_label.split("-")[0][1:])
            next_run_num = last_run_num + 1

        self.mlflow_config.run_name = (
            f"R{next_run_num:03d}-{datetime.now().strftime('%b%d-%H:%M')}"
        )

        return self


class HyperParameters(YamlBaseModel):
    learning_rate: float
    batch_size: int
    weight_decay: float
    num_epochs: int

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, TypeVar

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
    imagenet_dir: Annotated[
        Path, Field(default=".data/imagenet", validate_default=True)
    ]
    checkpoints: Annotated[
        Path, Field(default="src/solver/.logs/checkpoints", validate_default=True)
    ]
    tb_logs: Annotated[
        Path, Field(default="src/solver/.logs/tb_logs", validate_default=True)
    ]
    mlflow_uri: Annotated[
        str, Field(default="src/solver/.logs/mlflow_logs/mlflow", validate_default=True)
    ]
    jigsaw_dir: Annotated[Path, Field(default=".data/jigsaw", validate_default=True)]

    @field_validator(
        "imagenet_dir",
        "checkpoints",
        "tb_logs",
        "jigsaw_dir",
    )
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


class PiecemakerConfig(BaseModel):
    jigsaw_dir: Path = Field(
        default=None, description="Directory to store the files in"
    )
    number_of_pieces: int = Field(12, description="Target count of pieces")  # 3 x 4
    minimum_piece_size: int = Field(
        25,
        description="""Minimum piece size. Will change the count of pieces to
                        meet this if not set to 0.""",
    )
    maximum_piece_size: int = Field(
        512,
        description="""Maximum piece size. Will resize the image if not set
                        to 0 and should be at least greater than double the
                        set minimum piece size.""",
    )
    scaled_sizes: List[int] = Field(
        [100],
        description="""Comma separated list of sizes to scale for. Must
                        include 100 at least. Any that are too small will not
                        be created and a minimum scale will be done for the
                        ones that were dropped. Example: 33,68,100,150 for 4
                        scaled puzzles with the last one being at 150%.""",
    )
    use_max_size: bool = Field(
        False, description="Use the largest size when creating the size-100 directory"
    )
    variant: Literal["interlockingnubs", "stochasticnubs"] = Field(
        "interlockingnubs", description="Piece cut variant to use"
    )
    stochastic_nubs_probability: float = Field(0.5, description="Probability of nubs")
    gap: bool = Field(True, description="Leave gap between pieces")

    @model_validator(mode="after")
    def __check_options(self) -> "PiecemakerConfig":
        assert self.minimum_piece_size > 0
        assert self.number_of_pieces > 0
        assert self.minimum_piece_size > 1 and self.number_of_pieces > 1
        assert 100 in self.scaled_sizes

        # self.out_dir.mkdir(parents=True, exist_ok=True)
        # self.mask_dir.mkdir(parents=True, exist_ok=True)

        return self


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
    piecemaker_config: PiecemakerConfig = Field(default_factory=PiecemakerConfig)

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

        self.piecemaker_config.jigsaw_dir = self.paths.jigsaw_dir

        return self


class HyperParameters(YamlBaseModel):
    learning_rate: float
    batch_size: int
    weight_decay: float
    num_epochs: int

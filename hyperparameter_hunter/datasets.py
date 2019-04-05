##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


##################################################
# Dataset Chunks (Inputs, Targets, Predictions)
##################################################
@dataclass
class DatasetChunk:
    run: Optional[pd.DataFrame] = field(default=None, init=False)
    fold: Optional[pd.DataFrame] = field(default=None, init=False)
    rep: Optional[pd.DataFrame] = field(default=None, init=False)
    final: Optional[pd.DataFrame] = field(default=None, init=False)


@dataclass
class InputChunk(DatasetChunk):
    d: pd.DataFrame


@dataclass
class TargetChunk(DatasetChunk):
    d: pd.DataFrame


@dataclass
class PredictionChunk(DatasetChunk):
    ...


##################################################
# Dataset Kinds (Train, Validation, Holdout, Test)
##################################################
@dataclass
class BaseDataset:
    input: InputChunk
    target: TargetChunk
    prediction: PredictionChunk = field(default_factory=lambda: PredictionChunk(), init=False)

    def __post_init__(self):
        self.input = InputChunk(self.input)
        self.target = TargetChunk(self.target)


@dataclass
class Dataset(BaseDataset):
    # TODO: Update `wranglers`/`evaluators` to actually use `T` instead of old attributes
    T: Optional[BaseDataset] = field(default=None, init=False)
    # TODO: Update `wranglers`/`evaluators` to actually use `T` instead of old attributes


@dataclass
class TrainData(Dataset):
    prediction: None = field(default=None, init=False)


@dataclass
class OOFData(Dataset):
    ...


@dataclass
class HoldoutData(Dataset):
    ...


@dataclass
class TestData(Dataset):
    target: None = field(default=None, init=False)

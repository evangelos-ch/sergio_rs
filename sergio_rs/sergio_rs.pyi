import enum
from typing import Self, SupportsFloat

import numpy.typing as npt
import polars

class GRN:
    def __init__(self) -> None: ...
    def add_interaction(
        self, reg: Gene, tar: Gene, k: float, h: float | None, n: float
    ) -> None: ...
    def set_mrs(self) -> None: ...
    def ko_perturbation(self, gene_name: str, mr_profile: MrProfile) -> Self: ...

class Gene:
    def __init__(self, name: str, decay: float) -> None: ...

class Sim:
    def __init__(
        self,
        grn: GRN,
        num_cells: int,
        safety_iter: int,
        scale_iter: int,
        dt: float,
        noise_s: int,
        seed: int,
    ) -> None: ...
    def simulate(self, mr_profile: MrProfile) -> polars.DataFrame: ...

class MrProfile:
    @classmethod
    def from_random(
        cls,
        grn: GRN,
        num_cell_types: int,
        low_range: tuple[SupportsFloat, SupportsFloat],
        high_range: tuple[SupportsFloat, SupportsFloat],
        seed: int,
    ) -> MrProfile: ...

class NoiseSetting(enum.Enum):
    DS1 = enum.auto()
    DS2 = enum.auto()
    DS3 = enum.auto()
    DS4 = enum.auto()
    DS5 = enum.auto()
    DS6 = enum.auto()
    DS7 = enum.auto()
    DS8 = enum.auto()
    DS13 = enum.auto()
    DS14 = enum.auto()

def add_technical_noise(
    expr: npt.NDArray, setting: NoiseSetting, seed: int
) -> npt.NDArray: ...

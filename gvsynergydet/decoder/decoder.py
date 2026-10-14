from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn
from pathlib import Path
from typing import List, Union, Tuple



@dataclass
class DatasetCfgCommon:
    image_shape: List[int]
    background_color: List[float]
    cameras_are_circular: bool
    overfit_to_scene: Union[str, None]
    # overfit_to_scene: str | None
    # view_sampler: ViewSamplerCfg


@dataclass
class DatasetScannetCfg(DatasetCfgCommon):
    name: Literal["scannet", "replica"]
    roots: List[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    load_depth: bool = True
    near: float = 0.5
    far: float = 15.0

DatasetCfg = DatasetScannetCfg


DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]



@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Union[Float[Tensor, "batch view height width"], None]
    # depth: Float[Tensor, "batch view height width"] | None


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: Tuple[int, int],
        # depth_mode: DepthRenderingMode | None = None,
        depth_mode: Union[DepthRenderingMode, None] = None,
    ) -> DecoderOutput:
        pass

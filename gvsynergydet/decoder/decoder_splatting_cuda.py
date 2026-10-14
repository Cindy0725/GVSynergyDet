from dataclasses import dataclass
from typing import Literal, Union

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor


from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput, DatasetCfg


from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
from typing import Tuple



# @dataclass
# class Gaussians:
#     means: Float[Tensor, "batch gaussian dim"]
#     covariances: Float[Tensor, "batch gaussian dim dim"]
#     harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
#     opacities: Float[Tensor, "batch gaussian"]
#     scales: Float[Tensor, "batch gaussian 3"]
#     rotations: Float[Tensor, "batch gaussian 4"]

# @dataclass
# class DecoderSplattingCUDACfg:
#     name: Literal["splatting_cuda"]


# class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
#     background_color: Float[Tensor, "3"]

#     def __init__(
#         self,
#         cfg: DecoderSplattingCUDACfg,
#         dataset_cfg: DatasetCfg,
#     ) -> None:
#         super().__init__(cfg, dataset_cfg)
#         self.register_buffer(
#             "background_color",
#             torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
#             persistent=False,
#         )

#     def forward(
#         self,
#         gaussians: Gaussians,
#         extrinsics: Float[Tensor, "batch view 4 4"],
#         intrinsics: Float[Tensor, "batch view 3 3"],
#         near: Float[Tensor, "batch view"],
#         far: Float[Tensor, "batch view"],
#         image_shape: Tuple[int, int],
#         depth_mode: Union[DepthRenderingMode, None] = None,
#         no_color: bool = False,
#     ) -> DecoderOutput:
#         b, v, _, _ = extrinsics.shape
#         if not no_color:
#             color, depth = render_cuda(
#                 rearrange(extrinsics, "b v i j -> (b v) i j"),
#                 rearrange(intrinsics, "b v i j -> (b v) i j"),
#                 rearrange(near, "b v -> (b v)"),
#                 rearrange(far, "b v -> (b v)"),
#                 image_shape,
#                 repeat(self.background_color, "c -> (b v) c", b=b, v=v),
#                 repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
#                 repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
#                 repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
#                 repeat(gaussians.opacities, "b g -> (b v) g", v=v),
#                 repeat(gaussians.rotations, "b g r-> (b v) g r", v=v), 
#                 repeat(gaussians.scales, "b g s-> (b v) g s", v=v), 
#                 scale_invariant = False,
#                 use_sh = True,
#             )
#             color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
#             depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)
#             # if we set scale_invariant to False, maybe we need to remove the following line of code.
#             # depth = depth / 2
        
#         else:
#             color = None

#         return DecoderOutput(
#             color,
#             None
#             if depth_mode is None
#             else depth
#             # else self.render_depth(
#             #     gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
#             # ),
#         )

#     def render_depth(
#         self,
#         gaussians: Gaussians,
#         extrinsics: Float[Tensor, "batch view 4 4"],
#         intrinsics: Float[Tensor, "batch view 3 3"],
#         near: Float[Tensor, "batch view"],
#         far: Float[Tensor, "batch view"],
#         image_shape: Tuple[int, int],
#         mode: DepthRenderingMode = "depth",
#     ) -> Float[Tensor, "batch view height width"]:
#         b, v, _, _ = extrinsics.shape
#         result = render_depth_cuda(
#             rearrange(extrinsics, "b v i j -> (b v) i j"),
#             rearrange(intrinsics, "b v i j -> (b v) i j"),
#             rearrange(near, "b v -> (b v)"),
#             rearrange(far, "b v -> (b v)"),
#             image_shape,
#             repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
#             repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
#             repeat(gaussians.opacities, "b g -> (b v) g", v=v),
#             mode=mode,
#         )
#         return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: Tuple[int, int],
        depth_mode: Union[DepthRenderingMode, None] = None,
        no_color: bool = False,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        if not no_color:
            color, depth = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            )
            color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
            depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)
            depth = depth / 2
        
        else:
            color = None

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else depth
            # else self.render_depth(
            #     gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            # ),
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: Tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)

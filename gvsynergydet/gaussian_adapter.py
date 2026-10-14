from dataclasses import dataclass

import torch

from torch import Tensor, nn

import numpy as np
import cv2

from math import prod

from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int64

from math import isqrt

from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F



class GaussianParameterGenerator(nn.Module):
    def __init__(self, 
                 feature_dim: int,
                 sh_degree: int = 3,
                 scale_min: float = 0.01,  # Changed from 0.1
                 scale_max: float = 1.0):  # Changed from 10.0
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sh_degree = sh_degree
        
        # Improved initialization
        # self.scale_net = nn.Sequential(
        #     nn.Linear(feature_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )
        self.scale_net = nn.Linear(feature_dim, 3)
        self.rotation_net = nn.Linear(feature_dim, 4)
        self.opacity_net = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.d_sh = (sh_degree + 1) ** 2
        self.sh_net = nn.Linear(feature_dim, 3 * self.d_sh)
        
        # Better SH initialization
        self.register_buffer("sh_mask", torch.ones((self.d_sh,)))
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1)**2] = 0.5 * (0.5**degree)
            
        # Initialize opacities lower
        # nn.init.constant_(self.opacity_net[0].bias, -1.0)  # ~0.27 initial opacity

    def forward(self, fused_features: torch.Tensor, fused_positions: torch.Tensor):
        # Scale prediction
        scales = self.scale_net(fused_features)
        # scales = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(scales)
        # scales = torch.exp(scales)
        # scales = 0.01*torch.exp(scales)
        # scales = 0.01 * torch.sigmoid(scales)
        scales = 0.02 * torch.sigmoid(scales)

        
        # Rotation normalization with epsilon
        rotations = F.normalize(self.rotation_net(fused_features), p=2, dim=-1, eps=1e-6)
        
        # Covariance stability
        covariances = build_covariance(
            scales.clamp(min=1e-3), 
            rotations
        )
        
        return {
            'means': fused_positions,
            'scales': scales,
            'rotations': rotations,
            'opacities': self.opacity_net(fused_features),
            # 'sh_coeffs': self.sh_net(fused_features).view(-1, 3, self.d_sh) * self.sh_mask,
            'sh_coeffs': self.sh_net(fused_features).view(-1, 3, self.d_sh),
            'covariances': covariances
        }
    
    

# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        with torch.device(device):
            sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> Tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def sample_image_grid(
    shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> Tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def sample_training_rays(
    image: Float[Tensor, "batch view channel ..."],
    intrinsics: Float[Tensor, "batch view dim dim"],
    extrinsics: Float[Tensor, "batch view dim+1 dim+1"],
    num_rays: int,
) -> Tuple[
    Float[Tensor, "batch ray dim"],  # origins
    Float[Tensor, "batch ray dim"],  # directions
    Float[Tensor, "batch ray 3"],  # sampled color
]:
    device = extrinsics.device
    b, v, _, *grid_shape = image.shape

    # Generate all possible target rays.
    xy, _ = sample_image_grid(Tuple(grid_shape), device)
    origins, directions = get_world_rays(
        rearrange(xy, "... d -> ... () () d"),
        extrinsics,
        intrinsics,
    )
    origins = rearrange(origins, "... b v xy -> b (v ...) xy", b=b, v=v)
    directions = rearrange(directions, "... b v xy -> b (v ...) xy", b=b, v=v)
    pixels = rearrange(image, "b v c ... -> b (v ...) c")

    # Sample random rays.
    num_possible_rays = v * prod(grid_shape)
    ray_indices = torch.randint(num_possible_rays, (b, num_rays), device=device)
    batch_indices = repeat(torch.arange(b, device=device), "b -> b n", n=num_rays)

    return (
        origins[batch_indices, ray_indices],
        directions[batch_indices, ray_indices],
        pixels[batch_indices, ray_indices],
    )


def intersect_rays(
    origins_x: Float[Tensor, "*#batch 3"],
    directions_x: Float[Tensor, "*#batch 3"],
    origins_y: Float[Tensor, "*#batch 3"],
    directions_y: Float[Tensor, "*#batch 3"],
    eps: float = 1e-5,
    inf: float = 1e10,
) -> Float[Tensor, "*batch 3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast the rays so their shapes match.
    shape = torch.broadcast_shapes(
        origins_x.shape,
        directions_x.shape,
        origins_y.shape,
        directions_y.shape,
    )
    origins_x = origins_x.broadcast_to(shape)
    directions_x = directions_x.broadcast_to(shape)
    origins_y = origins_y.broadcast_to(shape)
    directions_y = directions_y.broadcast_to(shape)

    # Detect and remove batch elements where the directions are parallel.
    parallel = einsum(directions_x, directions_y, "... xyz, ... xyz -> ...") > 1 - eps
    origins_x = origins_x[~parallel]
    directions_x = directions_x[~parallel]
    origins_y = origins_y[~parallel]
    directions_y = directions_y[~parallel]

    # Stack the rays into (2, *shape).
    origins = torch.stack([origins_x, origins_y], dim=0)
    directions = torch.stack([directions_x, directions_y], dim=0)
    dtype = origins.dtype
    device = origins.device

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "r b i, r b j -> r b i j")
    n = n - torch.eye(3, dtype=dtype, device=device).broadcast_to((2, 1, 3, 3))

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "r b i j -> b i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "r b i j, r b j -> r b i")
    rhs = reduce(rhs, "r b i -> b i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    result = torch.linalg.lstsq(lhs, rhs).solution

    # Handle the case of parallel lines by setting depth to infinity.
    result_all = torch.ones(shape, dtype=dtype, device=device) * inf
    result_all[~parallel] = result
    return result_all


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def norm_np(v):
    normalized_v = v / (torch.sqrt(torch.sum(v**2)+1e-8)+1e-8)
    return normalized_v

class Create_from_depth_map():
    # This class is implemented via np package, so this can only handle batch_size=1.
    # If want to handle multi-batch_size, we need to implement this via torch package.
    def __init__(self, intrinsic_matrix, height=960, width=1280, depth_trunc=12.0):
        '''

        :param intrinsic_matrix: np.array size of (3,3)
        :param height:
        :param width:
        '''
        self.intrinsic_matrix = intrinsic_matrix
        self.focal_length = (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1])
        self.principal_point = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.height = height
        self.width = width
        self.depth_trunc = depth_trunc
        self.cam_pos_cam_coord = torch.tensor([0, 0, 0])
        i_coords, j_coords = torch.meshgrid(torch.range(0,height-1), torch.range(0,width-1), indexing='ij')
        coord_mat = torch.cat((j_coords[..., None], i_coords[..., None]), dim=-1)
        principal_point_mat = torch.tensor([self.principal_point[0], self.principal_point[1]])
        principal_point_mat = principal_point_mat.reshape(1, 1, 2)
        focal_length_mat = torch.tensor([self.focal_length[0], self.focal_length[1]])
        focal_length_mat = focal_length_mat.reshape(1, 1, 2)
        self.focal_length_mat = focal_length_mat.to(intrinsic_matrix.device)
        self.principal_point_mat = principal_point_mat.to(intrinsic_matrix.device) 
        self.one_mat = torch.ones((height, width, 1)).to(intrinsic_matrix.device) 
        self.cam_coord_part = ((coord_mat - principal_point_mat) / focal_length_mat).to(intrinsic_matrix.device)


    def project(self, depth_map, extrinsic_c2w_matrix):
        '''
        This for one image once.
        :param depth_map: np.array (H, W)
        :param extrinsic_c2w_matrix: np.array (4,4)
        :return:
        '''
        z_mat = depth_map.reshape(self.height, self.width, 1)
        masked_depth_map = torch.zeros_like(z_mat, device=z_mat.device)
        masked_depth_map[(z_mat < self.depth_trunc) & (z_mat > 0)] = 1
        point_mask = masked_depth_map 

        # do the computation
        cam_coord_xy = self.cam_coord_part * z_mat
        cam_coord_xyz = torch.cat([cam_coord_xy, z_mat], dim=-1)  # (h, w, 3)
        cam_coord_xyz1 = torch.cat([cam_coord_xyz, self.one_mat], dim=-1)  # (h, w, 4)
        # obtain the world_coordinate
        xyz1_cam_coord = cam_coord_xyz1.permute(2, 0, 1)  # (4, h, w)
        xyz1_cam_coord = xyz1_cam_coord.reshape(4, self.height * self.width)  # (4, h*w)
        world_coord_mat1 = torch.matmul(extrinsic_c2w_matrix, xyz1_cam_coord)  # (4, h*w)
        world_coord_mat = world_coord_mat1[:3, :]  # (3, h*w)
        world_coord_mat = world_coord_mat.T  # (h*w, 3)
        points_dir_world_coord = self.obtain_point_dir(cam_coord_xyz, extrinsic_c2w_matrix)  # (h*w, 3)

        # reshape the size of point_mask from (h, w, 1) to (h*w, 1)
        point_mask = point_mask.reshape(self.height*self.width, 1)
        
        post_xy_coords = torch.matmul(extrinsic_c2w_matrix.inverse(), world_coord_mat1)[:3]
        post_xy_coords = (post_xy_coords / post_xy_coords[2:3, :])[:2].permute(1, 0)
        post_xy_coords = post_xy_coords * self.focal_length_mat.reshape(1,2) + self.principal_point_mat

        return world_coord_mat

    def obtain_point_dir(self, cam_coord_xyz, extrinsic_c2w_m):
        h, w, _ = cam_coord_xyz.shape
        points_dir = cam_coord_xyz  # (h, w, 3)
        points_dir = points_dir.reshape(h*w, 3)
        points_dir_normed = norm_np(points_dir)
        points_dir_normed = points_dir_normed.T  # (3, h*w)
        # change the dir under world coordinates
        cam2world_mat3x3 = extrinsic_c2w_m[:3, :3]
        points_dir_world_coor = torch.matmul(cam2world_mat3x3, points_dir_normed)  # (3, h*w)
        return points_dir_world_coor.T  # (h*w, 3)

    def read_depth(self, filepath):
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000
        return depth_im


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    load_depth: bool = False



class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_gaussians,
        image_shape: Tuple[int, int],
        eps: float = 1e-8,
        fusion: bool = False,
        coords = None,
    ):
        device = extrinsics.device
        h, w = image_shape

        if not fusion:
            scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

            # Map scale features to valid scale range.
            scale_min = self.cfg.gaussian_scale_min
            scale_max = self.cfg.gaussian_scale_max
            scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
            scales = scales * depths[..., None] * multiplier[..., None]

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

            # Apply sigmoid to get valid colors.
            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

            # Create world-space covariance matrices.
            covariances = build_covariance(scales, rotations)
            c2w_rotations = extrinsics[..., :3, :3]
            covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        
        if coords is None:
            _, directions = get_world_rays(coordinates, extrinsics, intrinsics)
            means = torch.zeros_like(directions, device=directions.device)
            b, v = intrinsics.shape[:2]
            intrinsic = intrinsics[0,0].clone().view(3,3)
            intrinsic[:1,:] *= w
            intrinsic[1:2,:] *= h
            depth_to_pcd_now = Create_from_depth_map(intrinsic, height=h, width=w,
                                            depth_trunc=15)
            for i in range(b):
                for j in range(v):
                    pcd_w1_now = depth_to_pcd_now.project(depths[i,j].view(h,w), extrinsics[i,j].view(4,4))
                    means[i,j] = pcd_w1_now[:,None,None]
            if fusion:
                return means
            harmonics = rotate_sh(sh, c2w_rotations[..., None, :, :])
 
        else:
            harmonics = sh

        return Gaussians(
            means=means if (coords is None) else coords,
            covariances=covariances,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh

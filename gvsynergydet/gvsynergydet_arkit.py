# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
from einops import rearrange, reduce, repeat
from jaxtyping import Float
from .types import Gaussians


import torch

import random
import torch.nn as nn
from mmengine.structures import InstanceData
import numpy as np


from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList
from .depth_est_arkit import dinov2_vits14_lc_dpt
# the difference between depth_est_arkit and depth_est is only the image size, the model is the same, but the image size is different. The depth_est_arkit is for arkit dataset, the depth_est is for scannet dataset.  
from torch.nn import functional as F
from .gaussian_adapter import Create_from_depth_map, get_world_rays, sample_image_grid
from .gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from torch import Tensor, nn
from .decoder import DecoderSplattingCUDACfg, DatasetCfg, DecoderSplattingCUDA
from .decoder.decoder import DecoderOutput
from typing import List, Tuple



def knn(x, ref, k, maskself=False):
    '''
    x: (b,c,num_src)
    ref: (b, c, num_ref)
    k: top-k neigbour for each src
    assume x and ref are the same here!!
    '''
    
    inner = -2 * torch.matmul(x.transpose(2, 1), ref) #(B,num_src,num_ref)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) #(B,1,num_src)
    yy = torch.sum(ref ** 2, dim=1, keepdim=True) #(B,1,num_ref)
    
    pairwise_distance = -yy - inner - xx.transpose(2, 1) #(B, num_src, num_ref)
    
    # mask out self
    if maskself:
        assert x.shape == ref.shape
        mask = torch.arange(xx.shape[2]) # (num_src)
        pairwise_distance[:, mask, mask] = -100000
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B,N,k)
    return idx


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, maskself=False, angular_dist_method='dist',
                         scene_center=(0, 0, 0)):
        '''
        Args:
            tar_pose: target pose [num_tgt, 4, 4]. c2w
            ref_poses: reference poses [num_src, 4, 4]. C2W. we pick neighbour views from this pool
            num_select: the number of nearest views to select
        Returns: the selected indices # (num_tgt, k)
        '''
        num_cams = len(ref_poses)
        num_select = min(num_select, num_cams-1)
        # batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0) # (N, 3, 3)
        if angular_dist_method == 'dist':
            tar_cam_locs = tar_pose[:, :3, 3].unsqueeze(0).transpose(2, 1) # (1, 3, N)
            ref_cam_locs = ref_poses[:, :3, 3].unsqueeze(0).transpose(2, 1) # (1, 3, N)
            
            neigh_ids = knn(tar_cam_locs, ref_cam_locs, k=num_select, maskself=maskself)[0] # (num_src, k)

        return neigh_ids


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)






# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1) # 20, 4, 25600
    points_2d_3 = torch.bmm(projection.float(), points.float()) # projection: 20, 3, 4 points_2d_3: 20, 3, 25600
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long() # 20, 25600
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 20, 25600
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid

def _compute_projection(img_meta, stride):
    projection = []
    intrinsic = torch.tensor(img_meta['cam2img'][0][:3, :3])
    # ratio = img_meta['ori_shape'][0] / (img_meta['batch_input_shape'][0] / stride)

    ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0][0] / stride)
    ratio2 = img_meta['ori_shape'][1] / (img_meta['img_shape'][0][1] / stride)
    intrinsic[0] /= ratio2
    intrinsic[1] /= ratio
    # intrinsic[:2] /= ratio
    # use predicted pitch and roll for SUNRGBDTotal test

    extrinsics = map(torch.tensor, img_meta['lidar2cam'])
    for extrinsic in extrinsics:
        # projection.append(intrinsic @ extrinsic[:3])
        projection.append(intrinsic @ extrinsic.inverse()[:3, :])

    return torch.stack(projection)




def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts




@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()



class GRU(nn.Module):
    def __init__(self, input_channel=64, hidden_channel=64, weights_dim=24):
        super(GRU, self).__init__()
        self.mlp_z = nn.Sequential(nn.Linear(hidden_channel + input_channel + 2*weights_dim, hidden_channel),
                                   nn.ReLU(),
                                   nn.Linear(hidden_channel, hidden_channel))
        self.mlp_r = nn.Sequential(nn.Linear(hidden_channel + input_channel + 2*weights_dim, hidden_channel),
                                   nn.ReLU(),
                                   nn.Linear(hidden_channel, hidden_channel))
        self.mlp_n = nn.Sequential(nn.Linear(hidden_channel + input_channel + 1*weights_dim, hidden_channel),
                                   nn.ReLU(),
                                   nn.Linear(hidden_channel, hidden_channel))

    def forward(self, input_feat, hidden_feat, input_weights_emb, hidden_weights_emb):
        if len(input_feat.size()) == 2 and input_feat.size(0) == 1:
            input_feat = input_feat.unsqueeze(1)
        if hidden_feat is None:
            hidden_feat = torch.zeros_like(input_feat)
        input_feat_1 = torch.cat((input_feat, input_weights_emb), dim=-1) 
        hidden_feat_1 = torch.cat((hidden_feat, hidden_weights_emb), dim=-1)
        concat_input = torch.cat((hidden_feat_1, input_feat_1), dim=-1)  # B x N x C
        r = torch.sigmoid(self.mlp_r(concat_input))
        z = torch.sigmoid(self.mlp_z(concat_input))
        update_feat = torch.cat((r * hidden_feat, input_feat_1), dim=-1)
        q = torch.tanh(self.mlp_n(update_feat))
        output = (1 - z) * hidden_feat + z * q
        return output



class GaussianFeatureFusion(nn.Module):
    def __init__(self, gaussian_feat_dim=64, voxel_feat_dim=256):
        super().__init__()
        
        # Project Gaussian features to voxel feature dimension
        self.gaussian_proj = nn.Sequential(
            nn.Conv3d(gaussian_feat_dim, 128, 1),
            nn.ReLU(),
            nn.Conv3d(128, voxel_feat_dim, 1)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv3d(voxel_feat_dim * 2, voxel_feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(voxel_feat_dim, voxel_feat_dim, 3, padding=1),
            nn.ReLU()
        )
        
        # Adaptive weighting
        self.weight_net = nn.Sequential(
            nn.Conv3d(voxel_feat_dim * 2, 64, 1),
            nn.ReLU(),
            nn.Conv3d(64, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, voxel_feats, gaussian_features_voxelized, valids_gaussian=None):
        # Project Gaussian features
        gaussian_proj = self.gaussian_proj(gaussian_features_voxelized)
        
        # Apply validity mask if available
        if valids_gaussian is not None:
            gaussian_proj = gaussian_proj * valids_gaussian.float()
        
        # Concatenate for weighting
        concat = torch.cat([voxel_feats, gaussian_proj], dim=1)
        
        # Compute adaptive weights
        weights = self.weight_net(concat)
        w_voxel, w_gaussian = weights[:, 0:1], weights[:, 1:2]
        
        # Weighted combination
        weighted_voxel = voxel_feats * w_voxel
        weighted_gaussian = gaussian_proj * w_gaussian
        
        # Fuse
        combined = torch.cat([weighted_voxel, weighted_gaussian], dim=1)
        output = self.fusion(combined)
        
        # Residual connection
        # return output + voxel_feats
        return output

@MODELS.register_module()
class GVSynergyDet_ARKit(Base3DDetector):
    """`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior points
            generator config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 voxel_size: List,
                 coord_type: str,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck_3d = MODELS.build(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.feature_fusion = GaussianFeatureFusion(64, 256)
        self.depth_est = dinov2_vits14_lc_dpt(layers=1)

        self.opt = {'bg_color': [0, 0, 0], 'zfar': 100.0, 'znear': 0.01, 'trans': [0.0, 0.0, 0.0], 'scale': 1.0}       
        self.gaussian_regressor_v2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 66, 1)
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                64,
                36,
            ),
        )

        self.gru = GRU()
        self.decoder = DecoderSplattingCUDA(DecoderSplattingCUDACfg(name='splatting_cuda'), DatasetCfg(image_shape=[364, 560],cameras_are_circular=False,overfit_to_scene=False, name='scannet', roots='',baseline_epsilon=1e-3, max_fov=100.0, make_baseline_1=False, augment=True, background_color=[0.0, 0.0, 0.0]))
        self.sh_degree = 2
        self.d_sh = (self.sh_degree + 1) ** 2
        
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        # to do: maybe we need to change the voxel size and number of voxels；  
        self.gaussian_adapter = GaussianAdapter(GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max= 15.0, sh_degree = 2))

        self.voxel_size = voxel_size

        self.label_smoothing = 1.05

        self.epoch_n = 0
        self.inference_time = 0


    def predict_coords(self, depths, extrinsics, intrinsics, image_shape, gaussians_feats): 
        h, w = image_shape
        device = depths.device
        depths = rearrange(depths, "(b v) c h w -> b v (c h w) () ()", b=1)
        coordinates, _ = sample_image_grid((h, w), device)
        coordinates = rearrange(coordinates, "h w xy -> (h w) () xy")
        offset_xy = torch.zeros_like(rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=1), device=device) # (b, n_views, 196608, 1, 2)
        coordinates = coordinates + offset_xy # (b, num_context, num_gaussians, 1, 2)
        xy_ray = rearrange(coordinates, "b v r srf xy -> b v r srf () xy")
        _, directions = get_world_rays(xy_ray, extrinsics, intrinsics)
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
        
        return means, coordinates

    def fuse_gaussians_original(self, gaussians, coords, densities, weight_emb, depths, 
                       extrinsics, intrinsics, image_shape, depth_thres=0.1):
        # gaussians: a list with one element with shape (b, n_context, n_gaussians, 64)
        # coords: a list with one element with shape (b, n_context, n_gaussians, 1, 1, 3)
        # densities: a tensor with size (b, n_context, n_gaussians, 1, 1)
        # weights_emb: a tensor with size (b, n_context, n_gaussians, 1, 1)
        # depths: a tensor with size (n_context, 1, h, w)
        # extrinsics: a tensor with size (b, n_context, 4, 4)
        # intrinsics: a tensor with size (b, n_context, 3, 3)
        # image_shape: (h, w) = (384, 512)
        # depth_thres = 0.1
        length = gaussians[0].shape[1] # n_context
        global_gaussians = gaussians[0][:,0] # initialize the global_gaussians with the first view (b, n_gaussians, 64)
        global_densities = densities[:, 0] # initialize the global_densities with the first view (b, n_gaussians, 1, 1)
        global_weight_emb = weight_emb[:, 0] # initialize the global_weights with the first view (b, n_gaussians, 1, 1)
        global_coords = coords[0][:,0,:,0,0] # initialize the global_coords with the first view (b, n_gaussians, 3)
        global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians.shape[1],1,1) # initialize the global_extrinsics with the first view (b, n_gaussians, 4, 4)
        depths = rearrange(depths, "v c h w -> v (c h w)") # (n_context, n_gaussians)
        global_depths = depths[None, 0] # initialize the global_depths with the first view (1, n_gaussians)

        h, w = image_shape
        for i in range(1, length):
            extrinsic = extrinsics[0,i]
            intrinsic = intrinsics[0,i].clone()
            intrinsic[:1,:] *= w # better way than ours to scale the intrinsic matrix
            intrinsic[1:2,:] *= h
            focal_length = (intrinsic[0, 0], intrinsic[1, 1])
            principal_point = (intrinsic[0, 2], intrinsic[1, 2])
            principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device) # (1, 2)
            principal_point_mat = principal_point_mat.reshape(1, 2) # (1, 2)
            focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device) # (1, 2)
            focal_length_mat = focal_length_mat.reshape(1, 2) # (1, 2)
            means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, n_gaussians]
            post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3] # (n_gaussians, 2)
            curr_depths = post_xy_coords[2:3, :] # (n_gaussians)
            post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
            curr_depths = curr_depths.squeeze()
            post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
            pixel_coords = post_xy_coords.round().long()[:,[1,0]] # (n_gaussians, 2)
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0) # (n_gaussians)
            proj_map = - torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype)
            depth_map = torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype) * 10000

            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid] # shape equals to (n_valid) where n_valid is the valid indices. 
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin') # (n_gaussians) this is the global depth value for each gaussian

            fusion_mask = torch.abs(depth_map - depths[i]) < torch.clamp_min(depths[i] * 0.05, depth_thres) # (n_gaussians)

            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            if mask.sum() > 0:
                input_weights_emb = positional_encoding(torch.cat([global_densities[:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], global_weight_emb[:, mask]], dim=-1), 6)
                fusion_feat = self.gru(gaussians[0][:, i, pixel_indices][:,fusion_indices_].unsqueeze(2),
                                       global_gaussians[:, mask].unsqueeze(2),
                                       input_weights_emb,
                                       hidden_weights_emb).squeeze(2)

                global_gaussians = torch.cat([global_gaussians[:, ~mask], fusion_feat], dim=1)
                weights_0 = global_densities[:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)

                global_coords = torch.cat([global_coords[:, ~mask], (global_coords[:, mask]*weights_0[...,1] +
                                                coords[0][:, i, pixel_indices][:,fusion_indices_,0,0]*weights_1[...,1]) / (weights_0[...,1]+weights_1[...,1])], dim=1)
                global_densities = torch.cat([global_densities[:, ~mask], (global_densities[:, mask] +
                                                        densities[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                global_weight_emb = torch.cat([global_weight_emb[:, ~mask], (global_weight_emb[:, mask] +
                                                        weight_emb[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                
                global_extrinsics = torch.cat([global_extrinsics[:, ~mask], (global_extrinsics[:, mask]*weights_0[...,:1] +
                                                extrinsics[:, i, None]*weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1])], dim=1)
                global_depths = torch.cat([global_depths[:, ~mask], (global_depths[:, mask]*weights_0[...,0,0] +
                                                        depths[None, i, pixel_indices][:,fusion_indices_]*weights_1[...,0,0])/(weights_0[...,0,0]+weights_1[...,0,0])], dim=1)
            
            global_gaussians = torch.cat([global_gaussians, gaussians[0][:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_coords = torch.cat([global_coords, coords[0][:,i]\
                                                [:,~fusion_mask,0,0]], dim=1)
            
            global_densities = torch.cat([global_densities, densities[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_weight_emb = torch.cat([global_weight_emb, weight_emb[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_extrinsics = torch.cat([global_extrinsics, extrinsics[:,i,None].repeat(1,(~fusion_mask).sum(),1,1)], dim=1)
            global_depths = torch.cat([global_depths, depths[None,i]\
                                                [:,~fusion_mask]], dim=1)

        return global_gaussians, global_coords, global_extrinsics, global_depths

    
    def gaussian_encoder(self, gaussian_features, predicted_depth, context_images, extrinsics_context, intrinsics_context):
        b, n_views, _, h, w = context_images.shape
        gaussians = []
        coords = []
        results = {}

        densities = nn.Sigmoid()(rearrange(gaussian_features[:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views))
        # check the original code (activation function)
        weights = nn.Sigmoid()(rearrange(gaussian_features[:,1:2], "(b v) c h w -> b v (c h w) () ()", b=b)) # (b, n_views, 196608, 1, 1)
        gaussians_feats = rearrange(gaussian_features[:,2:], '(b v) c h w -> b v (h w) c', b=b, v=n_views)
        coord, xy_rays = self.predict_coords(predicted_depth, rearrange(extrinsics_context, "b v i j -> b v () () () i j"),
                rearrange(intrinsics_context, "b v i j -> b v () () () i j"), (h, w), gaussians_feats)
        coords.append(coord)
        gaussians.append(gaussians_feats)

        our_gaussians = []
        num_raw_gaussians = gaussians[0].shape[2] * gaussians[0].shape[1] # n_views * 196608
        B = gaussians[0].shape[0] # batch size
        for b in range(B):
            cur_gs = [x[b:b+1] for x in gaussians] # a list with one element with shape (b, n_context, n_gaussians, 64)
            cur_coords = [x[b:b+1] for x in coords] # a list with one element with shape (b, n_context, n_gaussians, 1, 1, 3)
            cur_densities = densities[b:b+1] # a tensor with size (b, n_context, n_gaussians, 1, 1)
            cur_weights = weights[b:b+1] # a tensor with size (b, n_context, n_gaussians, 1, 1)
            cur_depth = rearrange(predicted_depth, "(b v) c h w -> b v c h w", b=B)[b] # a tensor with size (n_context, 1, h, w)
            cur_gaussians, cur_coords, cur_extrinsics, cur_depths = self.fuse_gaussians_original(cur_gs, cur_coords, 
                                            cur_densities, cur_weights, 
                                            cur_depth, 
                                            extrinsics_context[b:b+1], \
                                            intrinsics_context[b:b+1], (h, w)) # cur_gaussians: (1, 545662, 64) cur_coords: (1, 545662, 3), cur_extrinsics: (1, 545662, 4, 4) cur_depths: (1, 545662)
            # gaussians_features = cur_gaussians.clone()
            gaussians_features = cur_gaussians
            cur_gaussians_now = rearrange(
                            self.to_gaussians(cur_gaussians),
                            "... (srf c) -> ... srf c",
                            srf=1,
                        ) # (1, 545662, 1, 36) 
            cur_gaussians = self.gaussian_adapter.forward(
                rearrange(cur_extrinsics, "b r i j -> b () r () () i j"),
                repeat(intrinsics_context[b:b+1,0], "b i j -> b () N () () i j", N=cur_gaussians_now.shape[1]),
                rearrange(xy_rays[b:b+1], "b v r srf xy -> b v r srf () xy"),
                rearrange(cur_depths, "b r -> b () r () ()"),
                nn.Sigmoid()(rearrange(cur_gaussians_now[..., :1], "b r srf c -> b () r srf c")),
                rearrange(cur_gaussians_now[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                fusion=False,
                coords=rearrange(cur_coords, "b r c -> b () r () () c"),
            ) # covariances: (1, 1, 545662, 1, 1, 3, 3) harmonics: (1, 1, 545662, 1, 1, 3, 9) means: (1, 1, 545662, 1, 1, 3) opacities: (1, 1, 545662, 1, 1) rotations: (1, 1, 545662, 1, 1, 4) scales: (1, 1, 545662, 1, 1, 3)
            our_gaussians.append(cur_gaussians)
        num_gaussians = our_gaussians[0].means.shape[2] # 545662
        results['gs_ratio'] = num_gaussians / num_raw_gaussians # 0.6938...
        gaussians = cur_gaussians
 
        results['num_gaussians'] = num_gaussians
        visualization_dump = {}
        try:
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
            gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
        except:
            visualization_dump["scales"] = gaussians.scales
            visualization_dump["rotations"] = gaussians.rotations
        
        results['visualizations'] = visualization_dump

        final_gs = []
        for i in range(len(our_gaussians)):
            final_gs.append(Gaussians(
                rearrange(
                    our_gaussians[i].means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    our_gaussians[i].covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    our_gaussians[i].harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    our_gaussians[i].opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            ))
        results['gaussians'] = final_gs

        return results, gaussians_features
    



    def voxelize_gaussians_pytorch_with_mask(self, means, features, voxel_size, scene_bounds):
        """
        Optimized voxelization of 3D Gaussian parameters using PyTorch with a validity mask for the voxels.

        Args:
            means (torch.Tensor): (num_gaussians, 3) tensor of Gaussian means (x, y, z).
            features (torch.Tensor): (num_gaussians, C) tensor of Gaussian features (e.g., covariances, opacities).
            voxel_size (float or list): Size of each voxel. Can be:
                - float: uniform voxel size across x, y, z
                - list/tuple of 3 floats: [voxel_size_x, voxel_size_y, voxel_size_z]
            scene_bounds (list or tuple): [xmin, xmax, ymin, ymax, zmin, zmax], specifying the 3D bounds of the scene.

        Returns:
            voxel_grid (torch.Tensor): 4D tensor of shape (Nx, Ny, Nz, C), where (Nx, Ny, Nz) are the grid dimensions
                                    and C is the feature dimension aggregated into voxels.
            valids (torch.Tensor): 4D tensor of shape (1, Nx, Ny, Nz), with 1 indicating voxels with at least one Gaussian,
                                and 0 otherwise.
        """

        # Scene bounds
        xmin, ymin, zmin, xmax, ymax, zmax = scene_bounds

        # Handle voxel_size as either float or list/tuple
        if isinstance(voxel_size, (list, tuple)):
            voxel_size_x, voxel_size_y, voxel_size_z = voxel_size
        else:
            voxel_size_x = voxel_size_y = voxel_size_z = voxel_size

        # Compute voxel grid dimensions
        Nx = int((xmax - xmin) / voxel_size_x) 
        Ny = int((ymax - ymin) / voxel_size_y) 
        Nz = int((zmax - zmin) / voxel_size_z) 

        # Create voxel size tensor for proper division
        voxel_size_tensor = torch.tensor([voxel_size_x, voxel_size_y, voxel_size_z], 
                                        device=means.device, dtype=means.dtype)

        # Normalize means to voxel indices (vectorized)
        voxel_indices = torch.floor((means - torch.tensor([xmin, ymin, zmin], device=means.device)) / voxel_size_tensor).long()

        # Filter out points outside the scene bounds
        valid_mask = (
            (0 <= voxel_indices[:, 0]) & (voxel_indices[:, 0] < Nx) &
            (0 <= voxel_indices[:, 1]) & (voxel_indices[:, 1] < Ny) &
            (0 <= voxel_indices[:, 2]) & (voxel_indices[:, 2] < Nz)
        )
        voxel_indices = voxel_indices[valid_mask]
        features = features[valid_mask]

        # Flatten the 3D voxel indices into a single dimension for efficient aggregation
        flattened_indices = voxel_indices[:, 0] * (Ny * Nz) + voxel_indices[:, 1] * Nz + voxel_indices[:, 2]

        # Aggregation: Sum features in each voxel
        C = features.shape[1]  # Number of feature channels
        total_voxels = Nx * Ny * Nz
        feature_sums = torch.zeros((total_voxels, C), dtype=features.dtype, device=features.device)
        feature_sums.index_add_(0, flattened_indices, features)

        # Count the number of Gaussians per voxel
        counts = torch.bincount(flattened_indices, minlength=total_voxels).float()

        # Create the valids mask: 1 if at least one Gaussian falls into the voxel, 0 otherwise
        valids = (counts > 0).view(1, Nx, Ny, Nz)

        # Reshape back to 3D grid
        voxel_grid = feature_sums.view(Nx, Ny, Nz, C)
        counts = counts.view(Nx, Ny, Nz)

        # Normalize features by counts to get the average (if needed)
        non_empty_mask = counts > 0  # Shape [Nx, Ny, Nz]
        voxel_grid[non_empty_mask] /= counts[non_empty_mask].unsqueeze(-1)  # Add a new dimension for broadcasting
        voxel_grid[~valids.squeeze(0).unsqueeze(-1).expand_as(voxel_grid)] = 0.0

        return voxel_grid, valids.float()
    

    
    def select_views(self, total_views, N_context, N_target):
        """
        Select context and target views such that each target view lies between two adjacent context views.
        
        Args:
            total_views: Total number of views available.
            N_context: Number of context views to select.
            N_target: Number of target views to select.

        Returns:
            context_views: Indices of selected context views.
            target_views: Indices of selected target views.
        """
        # Step 1: Select N_context views (randomly but sort them for adjacency)
        context_views = sorted(random.sample(range(total_views), N_context))
        
        # Step 2: Select N_target views (target views lie between adjacent context views)
        target_views = []
        for i in range(len(context_views) - 1):
            # Get the range between adjacent context views
            start = context_views[i] + 1
            end = context_views[i + 1] - 1
            
            # Select target views in this range
            if start <= end:  # Ensure there is space for target views
                target_views += random.sample(range(start, end + 1), 
                                            min(N_target // (N_context - 1), end - start + 1))
        
        # Handle remaining target views if needed
        remaining_targets = N_target - len(target_views)
        if remaining_targets > 0:
            # Try to add targets from unused views (excluding context views and already selected targets)
            unused_views = [i for i in range(total_views) if i not in context_views and i not in target_views]
            target_views += random.sample(unused_views, min(remaining_targets, len(unused_views)))
        
        return context_views, target_views


    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList, N_context, N_target, train=True):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """

        
        if train == False:
            total_views = batch_inputs_dict['imgs'].shape[1]
            # trained_views, target_views = self.select_views(total_views, N_context, N_target)
            target_views = [total_views-1]
            trained_views = [num for num in range(total_views) if num not in target_views]
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]

            img = batch_inputs_dict['imgs'][:,trained_views,:,:,:]
            img = img.reshape([-1] + list(img.shape)[2:]) # b*n_context, 3, H, W
            B, C, H, W = img.size()

            stride = 1
            axis_aligned_extrinsics = []
            for i in range(len(batch_img_metas[0]['lidar2cam'])):
                axis_aligned_extrinsics.append(np.linalg.inv(np.dot(batch_img_metas[0]['axis_align_matrix'], batch_img_metas[0]['lidar2cam'][i])))

            extrinsics_context = torch.tensor(np.array(axis_aligned_extrinsics)).unsqueeze(0)[:, trained_views, :, :].to(img.device).float()
            extrinsics_target = torch.tensor(np.array(axis_aligned_extrinsics)).unsqueeze(0)[:, target_views, :, :].to(img.device).float()

            intrinsics_context = torch.tensor(batch_img_metas[0]['cam2img']).unsqueeze(0)[:, trained_views, :3, :3].to(img.device).float() # (b=1, n_context, 3, 3)
            intrinsics_target = torch.tensor(batch_img_metas[0]['cam2img']).unsqueeze(0)[:, target_views, :3, :3].to(img.device).float()
            intrinsics_context[:, :, 0] /= batch_img_metas[0]['ori_shape'][1]
            intrinsics_context[:, :, 1] /= batch_img_metas[0]['ori_shape'][0]
            intrinsics_target[:, :, 0] /= batch_img_metas[0]['ori_shape'][1]
            intrinsics_target[:, :, 1] /= batch_img_metas[0]['ori_shape'][0]

            target_near = torch.full((1, len(target_views)), 0.5).to(img.device)
            target_far = torch.full((1, len(target_views)), 5.5).to(img.device)


            vision_features, predicted_depth, vision_features_gaussian = self.depth_est(img)

            predicted_depth = torch.clamp(predicted_depth, min=0.0, max=5.5)
            predicted_depth = nn.functional.interpolate(predicted_depth, size = [H, W], mode = 'bilinear')

            volumes, valids = [], []
            vision_features_new = vision_features
            stride = H / vision_features_new.shape[2]
            x = vision_features_new.reshape([1, -1] + list(vision_features_new.shape[1:]))
            for feature, img_meta in zip(x, batch_img_metas):
                projection = _compute_projection(img_meta, stride).to(x.device)[trained_views]
                points = get_points(
                    n_voxels=torch.tensor(self.n_voxels),
                    voxel_size=torch.tensor(self.voxel_size),
                    origin=torch.tensor(img_meta['origin'])
                ).to(x.device)

                height = int(img_meta['img_shape'][0][0] // stride)
                width = int(img_meta['img_shape'][0][1] // stride)         
                depth_maps = predicted_depth.permute(1,0,2,3).contiguous() # (1,20,416,640)
                depth_maps = nn.functional.interpolate(depth_maps, size = [H, W], mode = 'bilinear') # (1,20,img_height,img_width)
                volume, valid = backproject(feature[:, :, :height, :width], points, projection)
                volume = volume.sum(dim=0) #256,40,40,16         
                valid = valid.sum(dim=0) #1,40,40,16
                volume = volume / valid
                valid = valid > 0 #1,40,40,16 many falses
                volume[:, ~valid[0]] = .0 
                volumes.append(volume)
                valids.append(valid)
            
            x = torch.stack(volumes)
            valids = torch.stack(valids)

            gaussian_features = self.gaussian_regressor_v2(vision_features_gaussian)
            gaussian_features = nn.functional.interpolate(gaussian_features, size = [H, W], mode = 'bilinear') # (b, 65, h, w)
            encoder_results, gaussian_features_new = self.gaussian_encoder(gaussian_features, predicted_depth, img.unsqueeze(0), extrinsics_context, intrinsics_context)
            
            gaussians = encoder_results['gaussians']
            
            if not isinstance(gaussians, list):
                output = self.decoder.forward(
                    gaussians,
                    extrinsics_target,
                    intrinsics_target,
                    target_near,
                    target_far,
                    (H, W),
                    depth_mode=self.train_cfg.depth_mode,
                )
            else:
                output_list = []
                for i, gs in enumerate(gaussians):
                    output_list.append(self.decoder.forward(
                        gs,
                        extrinsics_target,
                        intrinsics_target,
                        target_near,
                        target_far,
                        (H, W),
                        depth_mode='depth',
                    ))
                output = DecoderOutput(None, None)
                output.color = torch.cat([x.color for x in output_list], dim=0)
                try:
                    output.depth = torch.cat([x.depth for x in output_list], dim=0)
                except:
                    pass


            gaussians_means = gaussians[0].means.clone().squeeze(0)

            origin = batch_img_metas[0]['origin']
            a, b, c = origin[0], origin[1], origin[2]

            voxelized_gaussian_features, valids_gaussian = self.voxelize_gaussians_pytorch_with_mask(gaussians_means, gaussian_features_new.squeeze(0), self.voxel_size, (-3.2+a, -3.2+b, -1.6+c, 3.2+a, 3.2+b, 1.6+c))
            voxelized_gaussian_features = voxelized_gaussian_features.unsqueeze(0).permute(0, 4, 1, 2, 3)
            x = self.feature_fusion(x, voxelized_gaussian_features, valids_gaussian)

            x = self.neck_3d(x)
            # valids: (1, 1, 40, 40, 16); target_occ: (1, 40, 40, 16); valids_gaussian: (1, 40, 40, 16)
            return output.color, output.depth, predicted_depth, target_views, trained_views, x, valids
        else:
            total_views = batch_inputs_dict['imgs'].shape[1]
            target_views = [total_views-1, total_views-2]
            context_views = [total_views-3, total_views-4, total_views-5, total_views-6]
            trained_views = [num for num in range(total_views) if num not in target_views and num not in context_views]
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            render_src_id = get_nearest_pose_ids(torch.tensor(np.array(batch_img_metas[0]['lidar2cam']))[target_views], torch.tensor(np.array(batch_img_metas[0]['lidar2cam']))[trained_views], num_select=1, maskself=False)
            # render_src_id is a tensor of shape (2,1)
            context_views.append(render_src_id[0].item())
            context_views.append(render_src_id[1].item())

            img_trained = batch_inputs_dict['imgs'][:,trained_views,:,:,:]
            img_context = batch_inputs_dict['imgs'][:,context_views,:,:,:]
            img_trained = img_trained.reshape([-1] + list(img_trained.shape)[2:]) # b*n_context, 3, H, W
            img_context = img_context.reshape([-1] + list(img_context.shape)[2:])
            _, C, H, W = img_trained.size() 
                          


            stride = 1
            axis_aligned_extrinsics = []

            # debug this part, calculate the extrinsics matrix by for loop 
            for i in range(len(batch_img_metas[0]['lidar2cam'])):
                axis_aligned_extrinsics.append(np.linalg.inv(np.dot(batch_img_metas[0]['axis_align_matrix'], batch_img_metas[0]['lidar2cam'][i])))
            extrinsics_context = torch.tensor(np.array(axis_aligned_extrinsics)).unsqueeze(0)[:, context_views, :, :].to(img_trained.device).float()
            extrinsics_target = torch.tensor(np.array(axis_aligned_extrinsics)).unsqueeze(0)[:, target_views, :, :].to(img_trained.device).float()
            extrinsics_trained = torch.tensor(np.array(axis_aligned_extrinsics)).unsqueeze(0)[:, trained_views, :, :].to(img_trained.device).float()
            

            intrinsics_context = torch.tensor(batch_img_metas[0]['cam2img']).unsqueeze(0)[:, context_views, :3, :3].to(img_trained.device).float() # (b=1, n_context, 3, 3)
            intrinsics_target = torch.tensor(batch_img_metas[0]['cam2img']).unsqueeze(0)[:, target_views, :3, :3].to(img_trained.device).float()
            intrinsics_trained = torch.tensor(batch_img_metas[0]['cam2img']).unsqueeze(0)[:, trained_views, :3, :3].to(img_trained.device).float()
            intrinsics_context[:, :, 0] /= batch_img_metas[0]['ori_shape'][1]
            intrinsics_context[:, :, 1] /= batch_img_metas[0]['ori_shape'][0]
            intrinsics_target[:, :, 0] /= batch_img_metas[0]['ori_shape'][1]
            intrinsics_target[:, :, 1] /= batch_img_metas[0]['ori_shape'][0]
            intrinsics_trained[:, :, 0] /= batch_img_metas[0]['ori_shape'][1]
            intrinsics_trained[:, :, 1] /= batch_img_metas[0]['ori_shape'][0]

            target_near = torch.full((1, len(target_views)), 0.5).to(img_trained.device)
            target_far = torch.full((1, len(target_views)), 5.5).to(img_trained.device)


            vision_features_trained, predicted_depth_trained, vision_features_trained_gaussian = self.depth_est(img_trained) #to be uncommented
            vision_features_context, predicted_depth_context, vision_features_context_gaussian = self.depth_est(img_context) #to be uncommented

            predicted_depth_trained = torch.clamp(predicted_depth_trained, min=0.0, max=5.5)
            predicted_depth_context = torch.clamp(predicted_depth_context, min=0.0, max=5.5)
            predicted_depth_trained = nn.functional.interpolate(predicted_depth_trained, size = [H, W], mode = 'bilinear')
            predicted_depth_context = nn.functional.interpolate(predicted_depth_context, size = [H, W], mode = 'bilinear')


            volumes, valids = [], []
            vision_features_new = vision_features_trained
            stride = H / vision_features_new.shape[2]
            x = vision_features_new.reshape([1, -1] + list(vision_features_new.shape[1:]))
            
            for feature, img_meta in zip(x, batch_img_metas):
                # use predicted pitch and roll for SUNRGBDTotal test
                # projection = self._compute_projection(img_meta, stride, angles).to(x.device)
                projection = _compute_projection(img_meta, stride).to(x.device)[trained_views]
                points = get_points(
                    n_voxels=torch.tensor(self.n_voxels),
                    voxel_size=torch.tensor(self.voxel_size),
                    origin=torch.tensor(img_meta['origin'])
                ).to(x.device)
                height = int(img_meta['img_shape'][0][0] // stride)
                width = int(img_meta['img_shape'][0][1] // stride)         
                volume, valid = backproject(feature[:, :, :height, :width], points, projection)
                volume = volume.sum(dim=0) #256,40,40,16         
                valid = valid.sum(dim=0) #1,40,40,16
                volume = volume / valid
                valid = valid > 0 #1,40,40,16 many falses
                volume[:, ~valid[0]] = .0 
                volumes.append(volume)
                valids.append(valid)
            
            x = torch.stack(volumes)
            valids = torch.stack(valids)

            gaussian_features_context = self.gaussian_regressor_v2(vision_features_context_gaussian)
            gaussian_features_trained = self.gaussian_regressor_v2(vision_features_trained_gaussian)
            gaussian_features_context = nn.functional.interpolate(gaussian_features_context, size = [H, W], mode = 'bilinear') # (b, 65, h, w)
            gaussian_features_trained = nn.functional.interpolate(gaussian_features_trained, size = [H, W], mode = 'bilinear') # (b, 65, h, w)
            encoder_results_context, gaussian_features_new_context= self.gaussian_encoder(gaussian_features_context, predicted_depth_context, img_context.unsqueeze(0), extrinsics_context, intrinsics_context)
            encoder_results_trained, gaussian_features_new_trained = self.gaussian_encoder(gaussian_features_trained, predicted_depth_trained, img_trained.unsqueeze(0), extrinsics_trained, intrinsics_trained)
            
            gaussians_context = encoder_results_context['gaussians']
            gaussians_trained = encoder_results_trained['gaussians']
            
            if not isinstance(gaussians_context, list):
                output = self.decoder.forward(
                    gaussians_context,
                    extrinsics_target,
                    intrinsics_target,
                    target_near,
                    target_far,
                    (H, W),
                    depth_mode=self.train_cfg.depth_mode,
                )
            else:
                output_list = []
                for i, gs in enumerate(gaussians_context):
                    output_list.append(self.decoder.forward(
                        gs,
                        extrinsics_target,
                        intrinsics_target,
                        target_near,
                        target_far,
                        (H, W),
                        depth_mode='depth',
                    ))
                output = DecoderOutput(None, None)
                output.color = torch.cat([x.color for x in output_list], dim=0)
                try:
                    output.depth = torch.cat([x.depth for x in output_list], dim=0)
                except:
                    pass
            gaussians_means = gaussians_trained[0].means.clone().squeeze(0)
            origin = batch_img_metas[0]['origin']
            a, b, c = origin[0], origin[1], origin[2]
            
            voxelized_gaussian_features, valids_gaussian = self.voxelize_gaussians_pytorch_with_mask(gaussians_means, gaussian_features_new_trained.squeeze(0), self.voxel_size, (-3.2+a, -3.2+b, -1.6+c, 3.2+a, 3.2+b, 1.6+c))
            voxelized_gaussian_features = voxelized_gaussian_features.unsqueeze(0).permute(0, 4, 1, 2, 3)
            x = self.feature_fusion(x, voxelized_gaussian_features, valids_gaussian)
            
            x = self.neck_3d(x)
            return output.color, output.depth, predicted_depth_trained, target_views, trained_views, x, valids
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        render_novel, render_depth, predicted_depth, target_views, trained_views, x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples, 50, 1, train=True)


        x += (valid_preds.float(), )
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)


        target_gt = batch_inputs_dict['imgs'][:, target_views, :, :, :]
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        mean_tensor = torch.tensor(mean, device=target_gt.device).view(1, 1, 3, 1, 1)  # (1,1,3,1,1)
        std_tensor = torch.tensor(std, device=target_gt.device).view(1, 1, 3, 1, 1)    # (1,1,3,1,1)
        denormalized = target_gt * std_tensor + mean_tensor
        target_gt = torch.clamp(denormalized, min=0.0, max=255.0) / 255.0 


        mseloss = nn.MSELoss()

        render_losses = mseloss(target_gt, render_novel)

        losses['render_loss'] = render_losses


        return losses
    
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        
        render_novel, render_depth, predicted_depth, target_views, trained_views, x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples, 100, 1, train=False)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        target_gt = batch_inputs_dict['imgs'][:, target_views, :, :, :]
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        mean_tensor = torch.tensor(mean, device=target_gt.device).view(1, 1, 3, 1, 1)  # (1,1,3,1,1)
        std_tensor = torch.tensor(std, device=target_gt.device).view(1, 1, 3, 1, 1)    # (1,1,3,1,1)
        denormalized = target_gt * std_tensor + mean_tensor
        target_gt = torch.clamp(denormalized, min=0.0, max=255.0) / 255.0 
        

        x += (valid_preds.float(), )
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)

        return predictions



    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        results = self.bbox_head.forward(x)
        return results

    def convert_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

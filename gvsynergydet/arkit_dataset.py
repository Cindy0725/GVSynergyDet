# Modified from
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/scannet_dataset.py
# Copyright (c) OpenMMLab, Inc. and its affiliates.
"""ARKitScenes Dataset
"""
from typing import Callable, List, Optional, Union
import numpy as np 

import cv2 
import warnings

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset


def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])



@DATASETS.register_module()
class AtlasARKitDataset(Det3DDataset):
    METAINFO = {
        'classes':
        ("cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", "toilet", 
         "bathtub", "oven", "dishwasher", "fireplace", "stool", "chair", "table", "tv_monitor", "sofa")}
    
    def __init__(self,                  
                 data_root: str,
                 ann_file: str, 
                 pipeline: List[Union[dict, Callable]] = [], 
                 modality: dict = dict(use_camera=False, use_lidar=True), 
                 classes=None, test_mode=False, num_frames=50, voxel_size=0.04, 
                 select_type='random', filter_empty_gt=True, metainfo: Optional[dict] = None, 
                 data_prefix: dict = dict(
                     pts='points',
                     pts_instance_mask='instance_mask',
                     pts_semantic_mask='semantic_mask'),
                     box_type_3d: str = 'Depth',
                     **kwargs)-> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        
        self.num_frames = num_frames
        self.voxel_size = voxel_size
        # self.data_infos = sorted(self.data_infos, key=lambda x: x['scene'])
        self.select_type = select_type

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)
    
    
    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)

        cam0_info = info.get('images', {}).get('CAM0', {})
        if isinstance(cam0_info.get('img_path'), list):
            # ARKit stores all frames of a scene as one list. The base
            # Det3DDataset camera parser assumes a single image path string.
            if not self.test_mode:
                info['ann_info'] = self.parse_ann_info(info)
            if self.test_mode and self.load_eval_anns:
                info['eval_ann_info'] = self.parse_ann_info(info)
        else:
            info = super().parse_data_info(info)

        return info
            

    
    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = super().parse_ann_info(info)
        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        # to target box structure

        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(ann_info['gt_bboxes_3d'], box_dim=ann_info['gt_bboxes_3d'].shape[-1], with_yaw=True, 
                                            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        return ann_info
    

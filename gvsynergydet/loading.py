import copy
import os.path as osp
from typing import Optional
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from mmdet3d.registry import TRANSFORMS


def _resolve_image_path(data_root: str, image_path: str) -> str:
    """Resolve relative annotation paths without changing absolute paths."""
    if '://' in image_path:
        return image_path
    return osp.join(data_root, image_path)



@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles3DGS(BaseTransform):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 num_views: int = 5,
                 num_ref_frames: int = -1,
                 nerf_target_views: int = 1,
                 loading: str = 'gap',
                 test_mode: bool = False,
                 set_default_scale: bool = True,
                 data_root: str = '',
                 load_depth: bool = False) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args
        self.num_views = num_views
        self.loading = loading
        self.nerf_target_views = nerf_target_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale
        self.data_root = data_root
        self.load_depth = load_depth

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape

        if self.loading == 'random':
            ids = np.arange(len(results['images']['CAM0']['img_path']))
            replace = True if self.num_views > len(ids) else False
            ids = np.random.choice(ids, self.num_views, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()
                target_id = target_id.tolist()

        elif self.loading == 'gap':
            ids = np.arange(len(results['images']['CAM0']['img_path']))
            src_1 = np.random.randint(0, len(ids)//2 - self.nerf_target_views//2 - 1, (1,))[0] # choose one from first half of images 
            src_3 = np.random.randint(len(ids)//2, len(ids)- self.nerf_target_views//2 - 1, (1,))[0]
            
            src_used_id = [src_1, src_1+self.nerf_target_views//2+1, src_3, src_3+self.nerf_target_views//2+1]
            target_id = []
            for k in range(self.nerf_target_views//2):
                target_id = target_id + [src_1+1+k, src_3+1+k]
            used_id = src_used_id + target_id

            replace = True if self.num_views > len(ids) else False
            rest_src = np.random.choice(np.setdiff1d(ids, np.array(used_id)), self.num_views-len(used_id), replace=replace)
            ids = rest_src.tolist() + src_used_id + target_id
            assert max(ids) < len(results['images']['CAM0']['img_path'])
        else:
            raise NotImplementedError(f'Loading type {self.loading} is not supported.')

        filename, cam2img, lidar2cam, depth_filename, lidar2cam_ori = [], [], [], [], []
        for i in ids:
            image_path = _resolve_image_path(
                self.data_root, results['images']['CAM0']['img_path'][i])
            filename.append(image_path)
            if self.load_depth:
                depth_filename.append(image_path.replace('jpg', 'png'))
            cam2img.append(results['images']['CAM0']['cam2img'])
            lidar2cam.append(results['images']['CAM0']['lidar2cam'][i])
            lidar2cam_ori.append(results['images']['CAM0']['lidar2cam_ori'][i])

        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam
        results['ori_lidar2cam'] = lidar2cam_ori


        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
        origin = np.array([.0, .0, .5])
        shift = np.random.normal(.0, (.7, .7, .0), 3)
        results['origin'] = origin + shift

        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]

        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]

        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        if self.load_depth:
            depths = [
                mmcv.imfrombytes(
                    get(name, backend_args=self.backend_args),
                    flag=self.color_type)
                for name in depth_filename
            ]
            depth = np.stack(depths, axis=-1) / 1000
            if self.to_float32:
                depth = depth.astype(np.float32)
            results['depth'] = [depth[..., i] for i in range(depth.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@TRANSFORMS.register_module()
class LoadMultiViewImageFromFilesARKit3DGS(BaseTransform):
    
    def __init__(self,
                to_float32: bool = False,
                color_type: str = 'unchanged',
                backend_args: Optional[dict] = None,
                num_views: int = 50,
                nerf_target_views: int = 1,
                loading: str = 'gap',
                test_mode: bool = False,
                set_default_scale: bool = True,
                data_root: str = '',
                load_depth: bool = False) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.loading = loading
        self.nerf_target_views = nerf_target_views
        self.backend_args = backend_args
        self.num_views = num_views
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale
        self.data_root = data_root
        self.load_depth = load_depth
    
    def transform(self, results: dict) -> Optional[dict]:

        if self.loading == 'random':
            ids = np.arange(len(results['images']['CAM0']['img_path']))
            replace = True if self.num_views > len(ids) else False
            ids = np.random.choice(ids, self.num_views, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()
                target_id = target_id.tolist()

        elif self.loading == 'gap':
            ids = np.arange(len(results['images']['CAM0']['img_path']))
            src_1 = np.random.randint(0, len(ids)//2 - self.nerf_target_views//2 - 1, (1,))[0] # choose one from first half of images 
            src_3 = np.random.randint(len(ids)//2, len(ids)- self.nerf_target_views//2 - 1, (1,))[0]
            
            src_used_id = [src_1, src_1+self.nerf_target_views//2+1, src_3, src_3+self.nerf_target_views//2+1]
            target_id = []
            for k in range(self.nerf_target_views//2):
                target_id = target_id + [src_1+1+k, src_3+1+k]
            used_id = src_used_id + target_id

            replace = True if self.num_views > len(ids) else False
            rest_src = np.random.choice(np.setdiff1d(ids, np.array(used_id)), self.num_views-len(used_id), replace=replace)
            ids = rest_src.tolist() + src_used_id + target_id
            assert max(ids) < len(results['images']['CAM0']['img_path'])
        else:
            raise NotImplementedError(f'Loading type {self.loading} is not supported.')

        filename, cam2img, lidar2cam, depth_filename = [], [], [], []
        # origins = []
        for i in ids:
            image_path = _resolve_image_path(
                self.data_root, results['images']['CAM0']['img_path'][i])
            filename.append(image_path)
            if self.load_depth:
                depth_path = image_path.replace('jpg', 'png')
                depth_path = depth_path.replace('lowres_wide', 'lowres_depth')
                depth_filename.append(depth_path)
            cam2img.append(results['images']['CAM0']['cam2img'][i])
            lidar2cam.append(results['images']['CAM0']['lidar2cam'][i])
            # origins.append(results['images']['CAM0']['lidar2cam'][i][])

        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
        origin = np.array([.0, .0, .5])
        shift = np.random.normal(.0, (.7, .7, .0), 3)
        results['origin'] = origin + shift

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]

        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        if self.load_depth:
            depths = [
                mmcv.imfrombytes(
                    get(name, backend_args=self.backend_args),
                    flag=self.color_type)
                for name in depth_filename
            ]
            depth = np.stack(depths, axis=-1) / 1000
            if self.to_float32:
                depth = depth.astype(np.float32)
            results['depth'] = [depth[..., i] for i in range(depth.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str

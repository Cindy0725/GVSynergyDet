"""Release regressions for RGB-only input and optional auxiliary depth."""
import copy

import cv2
import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmdet3d.structures import DepthInstance3DBoxes, Det3DDataSample
from projects.GVSynergyDet.gvsynergydet.formating import GVS_Pack3DDetInputs
from projects.GVSynergyDet.gvsynergydet.gvsynergydet_scannet import (
    GaussianFeatureFusion,
    GVSynergyDet_ScanNet,
)
from projects.GVSynergyDet.gvsynergydet.loading import (
    LoadMultiViewImageFromFiles3DGS,
    LoadMultiViewImageFromFilesARKit3DGS,
)
from projects.GVSynergyDet.gvsynergydet.processing import GVS_Resize


@pytest.fixture(params=['scannet', 'arkit'])
def scene(tmp_path, request):
    arkit = request.param == 'arkit'
    image_paths, depth_paths, extrinsics = [], [], []
    for index in range(12):
        if arkit:
            relative = f'Training/scene/scene_frames/lowres_wide/{index:05d}.png'
            depth_relative = relative.replace('lowres_wide', 'lowres_depth')
        else:
            relative = f'posed_images/scene/{index:05d}.jpg'
            depth_relative = relative.replace('.jpg', '.png')
        image_path = tmp_path / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((24, 32, 3), 40 + index, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        image_paths.append(relative)
        depth_paths.append(tmp_path / depth_relative)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index / 10
        extrinsics.append(pose)
    intrinsic = np.array([[25., 0., 15.], [0., 26., 11.], [0., 0., 1.]])
    camera = dict(img_path=image_paths, lidar2cam=extrinsics,
                  cam2img=[intrinsic.copy() for _ in image_paths]
                  if arkit else intrinsic)
    if not arkit:
        camera['lidar2cam_ori'] = copy.deepcopy(extrinsics)
    info = dict(
        images={'CAM0': camera}, axis_align_matrix=np.eye(4),
        gt_bboxes_3d=DepthInstance3DBoxes(
            [[0., 0., .5, 1., 1., 1.]], box_dim=6, with_yaw=False),
        gt_labels_3d=np.array([2], dtype=np.int64))
    loader = (LoadMultiViewImageFromFilesARKit3DGS if arkit
              else LoadMultiViewImageFromFiles3DGS)
    return tmp_path, info, depth_paths, loader


def pack(result):
    result = GVS_Resize(scale=(28, 20), keep_ratio=False)(result)
    return GVS_Pack3DDetInputs(
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])(result)


@pytest.mark.parametrize('absolute', [False, True], ids=['relative', 'absolute'])
@pytest.mark.parametrize('sampling', ['gap', 'random'])
def test_rgb_only_pipeline(scene, absolute, sampling):
    root, info, depth_paths, loader = scene
    if absolute:
        info['images']['CAM0']['img_path'] = [
            str(root / path) for path in info['images']['CAM0']['img_path']]
    result = loader(data_root=str(root), num_views=8,
                    nerf_target_views=2 if sampling == 'gap' else 1,
                    loading=sampling)(info)
    expected_views = 8 if sampling == 'gap' else 7
    assert all(path.startswith(str(root)) for path in result['filename'])
    assert 'depth' not in result and 'tsdf' not in result
    if sampling == 'random':
        frame_ids = [int(path.rsplit('/', 1)[-1].split('.')[0])
                     for path in result['filename']]
        assert frame_ids == sorted(frame_ids)
    packed = pack(result)
    assert packed['inputs']['img'].shape == (expected_views, 3, 20, 28)
    sample = packed['data_samples']
    assert len(sample.metainfo['cam2img']) == expected_views
    assert len(sample.metainfo['lidar2cam']) == expected_views
    assert np.asarray(sample.metainfo['origin']).shape == (3,)
    assert np.asarray(sample.metainfo['origin']).dtype == np.float64
    assert 'depth' not in sample.metainfo
    assert 'tsdf_vol' not in sample.metainfo
    assert 'tsdf_origin' not in sample.metainfo
    assert sample.gt_instances_3d.labels_3d.tolist() == [2]
    torch.testing.assert_close(sample.gt_instances_3d.bboxes_3d.tensor,
                               info['gt_bboxes_3d'].tensor)
    assert not any(path.exists() for path in depth_paths)
    assert not (root / 'atlas_tsdf').exists()


@pytest.mark.parametrize('sampling', ['gap', 'random'])
def test_optional_depth_without_tsdf(scene, sampling):
    root, info, depth_paths, loader = scene
    for path in depth_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.full((24, 32), 1500, dtype=np.uint16))
    result = loader(data_root=str(root), load_depth=True, num_views=8,
                    nerf_target_views=2 if sampling == 'gap' else 1,
                    loading=sampling)(info)
    expected_views = 8 if sampling == 'gap' else 7
    packed = pack(result)
    assert packed['inputs']['img'].shape == (expected_views, 3, 20, 28)
    depth = packed['data_samples'].metainfo['depth']
    torch.testing.assert_close(
        depth, torch.full((expected_views, 24, 32), 1.5))
    assert 'tsdf_vol' not in packed['data_samples'].metainfo
    assert not (root / 'atlas_tsdf').exists()


def test_scannet_predict_scores_rgb_without_gt_depth():
    class Head:
        def __call__(self, features):
            return (features, features, features)

        def get_bboxes(self, *args):
            return [InstanceData()]

    class PredictHarness:
        predict = GVSynergyDet_ScanNet.predict
        add_pred_to_datasample = GVSynergyDet_ScanNet.add_pred_to_datasample
        bbox_head = Head()

        def extract_feat(self, *args, **kwargs):
            rendered_rgb = torch.full((1, 1, 3, 16, 16), .45)
            return (rendered_rgb, None, None, [2], [0, 1],
                    torch.zeros(1), torch.ones(1))

        def _save_fusion_stats_jsonl(self, *args, **kwargs):
            return None

    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
    images = (torch.full((1, 3, 3, 16, 16), 127.5) - mean) / std
    sample = Det3DDataSample(metainfo={'img_path': ['posed_images/scene/0.jpg']})
    prediction = PredictHarness().predict({'imgs': images}, [sample])[0]
    assert 'pred_instances_3d' in prediction
    assert 'depth' not in sample.metainfo
    assert np.isfinite(prediction.psnr) and np.isfinite(prediction.ssim)
    assert prediction.psnr == pytest.approx(-10 * np.log10(.05 ** 2), abs=1e-4)
    assert 0 < prediction.ssim <= 1


def test_fusion_contains_only_the_full_model_path():
    torch.manual_seed(7)
    fusion = GaussianFeatureFusion(record_stats=False)
    voxel = torch.randn(1, 256, 3, 4, 2)
    gaussian = torch.randn(1, 64, 3, 4, 2)
    occupancy = torch.rand(1, 3, 4, 2) > .4

    gaussian_projected = fusion.gaussian_proj(gaussian)
    gaussian_masked = gaussian_projected * occupancy[:, None].float()
    weights = fusion.weight_net(torch.cat([voxel, gaussian_masked], dim=1))
    expected = fusion.fusion(torch.cat([
        voxel * weights[:, 0:1],
        gaussian_masked * weights[:, 1:2],
    ], dim=1))
    actual = fusion(voxel, gaussian, occupancy)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(TypeError):
        GaussianFeatureFusion(fusion_mode='voxel_only')

from typing import List, Tuple

import torch
from mmcv.cnn import Scale
from mmcv.ops import nms3d, nms3d_normal
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule, bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d.utils import rotation_3d_in_axis
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)


from .gvsynergydet_scannet import get_points
from mmdet3d.models.layers.box3d_nms import aligned_3d_nms
from mmdet3d.structures import DepthInstance3DBoxes


class ImVoxelHeadV2(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 n_reg_outs,
                 n_scales,
                 limit,
                 voxel_size,
                 centerness_topk=-1,
                 center_loss: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                 bbox_loss: ConfigType = dict(type='RotatedIoU3DLoss'),
                 cls_loss: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25),
                 train_cfg=None,
                 test_cfg=None):
        super(ImVoxelHeadV2, self).__init__()
        self.n_classes = n_classes
        self.n_scales = n_scales
        self.limit = limit
        self.centerness_topk = centerness_topk
        self.center_loss = MODELS.build(center_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.voxel_size = voxel_size
        self._init_layers(n_channels, n_reg_outs)

    def _init_layers(self, n_channels, n_reg_outs):
        self.centerness_conv = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)
        self.reg_conv = nn.Conv3d(n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.cls_conv = nn.Conv3d(n_channels, self.n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(self.n_scales)])

    # Follow AnchorFreeHead.init_weights
    def init_weights(self):
        normal_init(self.centerness_conv, std=.01)
        normal_init(self.reg_conv, std=.01)
        normal_init(self.cls_conv, std=.01, bias=bias_init_with_prob(.01))

    def forward(self, x):
        return multi_apply(self.forward_single, x, self.scales)

    def forward_train(self, x, valid, img_metas, gt_bboxes, gt_labels):
        loss_inputs = self(x) + (valid, img_metas, gt_bboxes, gt_labels)
        losses = self.loss(*loss_inputs)
        return losses

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             valid,
             img_metas,
             gt_bboxes,
             gt_labels):
        """
        Args:
            centernesses (list(Tensor)): Multi-level centernesses
                of shape (batch, 1, nx[i], ny[i], nz[i])
            bbox_preds (list(Tensor)): Multi-level xyz min and max distances
                of shape (batch, 6, nx[i], ny[i], nz[i])
            cls_scores (list(Tensor)): Multi-level class scores
                of shape (batch, n_classes, nx[i], ny[i], nz[i])
            img_metas (list[dict]): Meta information of each image
            gt_bboxes (list(BaseInstance3DBoxes)): Ground truth bboxes for each image
            gt_labels (list(Tensor)): Ground truth class labels for each image

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) == \
               len(valid) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        valids = []
        for x in centernesses:
            valids.append(nn.Upsample(size=x.shape[-3:], mode='trilinear')(valid).round().bool())

        loss_centerness, loss_bbox, loss_cls = [], [], []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                valids=[x[i] for x in valids],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            loss_centerness=torch.mean(torch.stack(loss_centerness)),
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls))
        )

    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     valids,
                     img_meta,
                     gt_bboxes,
                     gt_labels):
        """
        Args:
            centernesses (list(Tensor)): Multi-level centernesses
                of shape (1, nx[i], ny[i], nz[i])
            bbox_preds (list(Tensor)): Multi-level xyz min and max distances
                of shape (6, nx[i], ny[i], nz[i])
            cls_scores (list(Tensor)): Multi-level class scores
                of shape (n_classes, nx[i], ny[i], nz[i])
            img_metas (list[dict]): Meta information
            gt_bboxes (BaseInstance3DBoxes): Ground truth bboxes
                of shape (n_boxes, 7)
            gt_labels (list(Tensor)): Ground truth class labels
                of shape (n_boxes,)

        Returns:
            tuple(Tensor): 3 losses
        """
        featmap_sizes = [featmap.size()[-3:] for featmap in centernesses]
        mlvl_points = self.get_points(
            featmap_sizes=featmap_sizes,
            origin=img_meta['origin'],
            device=gt_bboxes.device
        )

        centerness_targets, bbox_targets, labels = self.get_targets(mlvl_points, gt_bboxes, gt_labels)

        flatten_centerness = [centerness.permute(1, 2, 3, 0).reshape(-1)
                              for centerness in centernesses]
        bbox_pred_size = bbox_preds[0].shape[0]
        flatten_bbox_preds = [bbox_pred.permute(1, 2, 3, 0).reshape(-1, bbox_pred_size)
                              for bbox_pred in bbox_preds]
        flatten_cls_scores = [cls_score.permute(1, 2, 3, 0).reshape(-1, self.n_classes)
                              for cls_score in cls_scores]
        flatten_valids = [valid.permute(1, 2, 3, 0).reshape(-1)
                          for valid in valids]

        flatten_centerness = torch.cat(flatten_centerness)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_valids = torch.cat(flatten_valids)

        flatten_centerness_targets = centerness_targets.to(centernesses[0].device)
        flatten_bbox_targets = bbox_targets.to(centernesses[0].device)
        flatten_labels = labels.to(centernesses[0].device)
        flatten_points = torch.cat(mlvl_points)

        # skip background
        pos_inds = torch.nonzero(torch.logical_and(
            flatten_labels >= 0,
            flatten_valids
        )).reshape(-1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centernesses[0].device)
        n_pos = max(reduce_mean(n_pos), 1.)
        if torch.any(flatten_valids):
            loss_cls = self.cls_loss(
                flatten_cls_scores[flatten_valids],
                flatten_labels[flatten_valids],
                avg_factor=n_pos
            )
        else:
            loss_cls = flatten_cls_scores[flatten_valids].sum()
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_preds = flatten_bbox_preds[pos_inds]

        if len(pos_inds) > 0:
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds].to(pos_bbox_preds.device)
            loss_centerness = self.center_loss(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.bbox_loss(
                self._bbox_pred_to_loss(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum()
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        return loss_centerness, loss_bbox, loss_cls

    @torch.no_grad()
    def get_points(self, featmap_sizes, origin, device):
        mlvl_points = []
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(get_points(
                n_voxels=torch.tensor(featmap_size),
                voxel_size=torch.tensor(self.voxel_size) * (2 ** i),
                origin=torch.tensor(origin)
            ).reshape(3, -1).transpose(0, 1).to(device))
        return mlvl_points

    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   valid,
                   img_metas):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(img_metas)
        valids = []
        for x in centernesses:
            valids.append(nn.Upsample(size=x.shape[-3:], mode='trilinear')(valid).round().bool())
        n_levels = len(centernesses)
        result_list = []
        for img_id in range(len(img_metas)):
            centerness_list = [
                centernesses[i][img_id].detach() for i in range(n_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(n_levels)
            ]
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(n_levels)
            ]
            valid_list = [
                valids[i][img_id].detach() for i in range(n_levels)
            ]
            det_bboxes_3d = self._get_bboxes_single(
                centerness_list, bbox_pred_list, cls_score_list, valid_list, img_metas[img_id]
            )
            result_list.append(det_bboxes_3d)
        return result_list

    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           valids,
                           img_meta):
        featmap_sizes = [featmap.size()[-3:] for featmap in centernesses]
        mlvl_points = self.get_points(
            featmap_sizes=featmap_sizes,
            origin=img_meta['origin'],
            device=centernesses[0].device
        )
        bbox_pred_size = bbox_preds[0].shape[0]
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, valid, points in zip(
            centernesses, bbox_preds, cls_scores, valids, mlvl_points
        ):
            centerness = centerness.permute(1, 2, 3, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 3, 0).reshape(-1, bbox_pred_size)
            scores = cls_score.permute(1, 2, 3, 0).reshape(-1, self.n_classes).sigmoid()
            valid = valid.permute(1, 2, 3, 0).reshape(-1)
            scores = scores * centerness[:, None] * valid[:, None]
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                points = points[ids]

            bboxes = self._bbox_pred_to_result(points, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        # return bboxes, scores, labels
        return results

    def forward_single(self, x, scale):
        raise NotImplementedError

    def _bbox_pred_to_loss(self, points, bbox_preds):
        raise NotImplementedError

    def _bbox_pred_to_result(self, points, bbox_preds):
        raise NotImplementedError

    def get_targets(self, points, gt_bboxes, gt_labels):
        raise NotImplementedError

    def _nms(self, bboxes, scores, img_meta):
        raise NotImplementedError


@MODELS.register_module()
class ImVoxelHead_ScanNet_V2(ImVoxelHeadV2):
    def forward_single(self, x, scale):
        return (
            self.centerness_conv(x),
            torch.exp(scale(self.reg_conv(x))),
            self.cls_conv(x)
        )
    

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    def _bbox_pred_to_result(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    @torch.no_grad()
    def get_targets(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        expanded_scales = [
            points[i].new_tensor(i).expand(len(points[i])).to(gt_labels.device)
            for i in range(len(points))
        ]
        points = torch.cat(points, dim=0).to(gt_labels.device)
        scales = torch.cat(expanded_scales, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:6]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        dx_min = expanded_points[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - expanded_points[..., 0]
        dy_min = expanded_points[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - expanded_points[..., 1]
        dz_min = expanded_points[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - expanded_points[..., 2]
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

        # condition2: positive points per scale >= limit
        # calculate positive points per scale
        n_pos_points_per_scale = []
        for i in range(self.n_scales):
            n_pos_points_per_scale.append(torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
        # find best scale
        n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0)
        lower_limit_mask = n_pos_points_per_scale < self.limit
        # fix nondeterministic argmax for torch<1.7
        extra = torch.arange(self.n_scales, 0, -1).unsqueeze(1).expand(self.n_scales, n_boxes).to(
            lower_limit_mask.device)
        lower_index = torch.argmax(lower_limit_mask.int() * extra, dim=0) - 1
        lower_index = torch.where(lower_index < 0, torch.zeros_like(lower_index), lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask), dim=0)
        best_scale = torch.where(all_upper_limit_mask, torch.ones_like(all_upper_limit_mask) * self.n_scales - 1,
                                 lower_index)
        # keep only points with best scale
        best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
        scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
        inside_best_scale_mask = best_scale == scales

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_best_scale_mask, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, self.centerness_topk + 1, dim=0).values[-1]
        inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_best_scale_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness_mask, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, self._bbox_pred_to_bbox(points, bbox_targets), labels

    def _nms(self, bboxes, scores, img_meta):
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = aligned_3d_nms(bboxes, scores, labels, self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        bboxes = torch.stack((
            (bboxes[:, 0] + bboxes[:, 3]) / 2.,
            (bboxes[:, 1] + bboxes[:, 4]) / 2.,
            (bboxes[:, 2] + bboxes[:, 5]) / 2.,
            bboxes[:, 3] - bboxes[:, 0],
            bboxes[:, 4] - bboxes[:, 1],
            bboxes[:, 5] - bboxes[:, 2]
        ), dim=1)
        # bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5), box_dim=6, with_yaw=False)
        bboxes = DepthInstance3DBoxes(bboxes, origin=(.5, .5, .5), box_dim=6, with_yaw=False)
        return bboxes, scores[ids], labels[ids]

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        return torch.stack([
            points[:, 0] - bbox_pred[:, 0],
            points[:, 1] - bbox_pred[:, 2],
            points[:, 2] - bbox_pred[:, 4],
            points[:, 0] + bbox_pred[:, 1],
            points[:, 1] + bbox_pred[:, 3],
            points[:, 2] + bbox_pred[:, 5]
        ], -1)


def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    # todo: sqrt ?
    return torch.sqrt(centerness_targets)


@MODELS.register_module()
class ImVoxelHead_ARKit(BaseModule):
    ''' directly copy from imvoxelhead in mmdet3d'''
    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 prior_generator: ConfigType,
                 center_loss: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                 bbox_loss: ConfigType = dict(type='RotatedIoU3DLoss'),
                 cls_loss: ConfigType = dict(type='mmdet.FocalLoss'),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super(ImVoxelHead_ARKit, self).__init__(init_cfg)
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.center_loss = MODELS.build(center_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        """Initialize neural network layers of the head."""
        self.conv_center = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)
        self.conv_reg = nn.Conv3d(
            n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.conv_cls = nn.Conv3d(n_channels, n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_levels)])

    def init_weights(self):
        """Initialize all layer weights."""
        normal_init(self.conv_center, std=.01)
        normal_init(self.conv_reg, std=.01)
        normal_init(self.conv_cls, std=.01, bias=bias_init_with_prob(.01))

    def _forward_single(self, x: Tensor, scale: Scale):
        """Forward pass per level.

        Args:
            x (Tensor): Per level 3d neck output tensor.
            scale (mmcv.cnn.Scale): Per level multiplication weight.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification predictions.
        """
        reg_final = self.conv_reg(x)
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)
        return self.conv_center(x), bbox_pred, self.conv_cls(x)

    def forward(self, x: Tensor):
        """Forward function.

        Args:
            x (list[Tensor]): Features from 3d neck.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification predictions.
        """
        return multi_apply(self._forward_single, x, self.scales)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        valid_pred = x[-1]
        outs = self(x[:-1])

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (valid_pred, batch_gt_instances_3d,
                              batch_input_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        valid_pred = x[-1]
        outs = self(x[:-1])
        predictions = self.predict_by_feat(
            *outs,
            valid_pred=valid_pred,
            batch_input_metas=batch_input_metas,
            rescale=rescale)
        return predictions

    def _loss_by_feat_single(self, center_preds, bbox_preds, cls_preds,
                             valid_preds, input_meta, gt_bboxes, gt_labels):
        """Per scene loss function.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            valid_preds (list[Tensor]): Valid mask predictions for all levels.
            input_meta (dict): Scene meta info.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor]: Centerness, bbox, and classification loss values.
        """
        featmap_sizes = [featmap.size()[-3:] for featmap in center_preds]
        points = self._get_points(
            featmap_sizes=featmap_sizes,
            # origin=input_meta['lidar2img']['origin'],
            origin=input_meta['origin'],
            device=gt_bboxes.device)
        
        center_targets, bbox_targets, cls_targets = self._get_targets(
            points, gt_bboxes, gt_labels)

        center_preds = torch.cat(
            [x.permute(1, 2, 3, 0).reshape(-1) for x in center_preds])
        bbox_preds = torch.cat([
            x.permute(1, 2, 3, 0).reshape(-1, x.shape[0]) for x in bbox_preds
        ])
        cls_preds = torch.cat(
            [x.permute(1, 2, 3, 0).reshape(-1, x.shape[0]) for x in cls_preds])
        valid_preds = torch.cat(
            [x.permute(1, 2, 3, 0).reshape(-1) for x in valid_preds])
        points = torch.cat(points)

        # cls loss
        pos_inds = torch.nonzero(
            torch.logical_and(cls_targets >= 0, valid_preds)).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.)
        if torch.any(valid_preds):
            cls_loss = self.cls_loss(
                cls_preds[valid_preds],
                cls_targets[valid_preds],
                avg_factor=n_pos)
        else:
            cls_loss = cls_preds[valid_preds].sum()

        # bbox and centerness losses
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        if len(pos_inds) > 0:
            pos_center_targets = center_targets[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_points = points[pos_inds]
            center_loss = self.center_loss(
                pos_center_preds, pos_center_targets, avg_factor=n_pos)
            bbox_loss = self.bbox_loss(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_center_targets,
                avg_factor=pos_center_targets.sum())
        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()
        return center_loss, bbox_loss, cls_loss

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     valid_pred: Tensor,
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:
        """Per scene loss function.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: Centerness, bbox, and classification loss values.
        """
        valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        center_losses, bbox_losses, cls_losses = [], [], []
        for i in range(len(batch_input_metas)):
            center_loss, bbox_loss, cls_loss = self._loss_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                valid_preds=[x[i] for x in valid_preds],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d)
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)))

    def _predict_by_feat_single(self, center_preds: List[Tensor],
                                bbox_preds: List[Tensor],
                                cls_preds: List[Tensor],
                                valid_preds: List[Tensor],
                                input_meta: dict) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            valid_preds (tuple[Tensor]): Upsampled valid masks for all feature
                levels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """
        featmap_sizes = [featmap.size()[-3:] for featmap in center_preds]
        points = self._get_points(
            featmap_sizes=featmap_sizes,
            # origin=input_meta['lidar2img']['origin'],
            origin=input_meta['origin'],
            device=center_preds[0].device)
        
        mlvl_bboxes, mlvl_scores = [], []
        for center_pred, bbox_pred, cls_pred, valid_pred, point in zip(
                center_preds, bbox_preds, cls_preds, valid_preds, points):
            center_pred = center_pred.permute(1, 2, 3, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 3,
                                          0).reshape(-1, bbox_pred.shape[0])
            cls_pred = cls_pred.permute(1, 2, 3,
                                        0).reshape(-1, cls_pred.shape[0])
            valid_pred = valid_pred.permute(1, 2, 3, 0).reshape(-1, 1)

            scores = cls_pred.sigmoid() * center_pred.sigmoid() * valid_pred
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=bboxes.shape[1],
            with_yaw=bboxes.shape[1] == 7,
            origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                        bbox_preds: List[List[Tensor]],
                        cls_preds: List[List[Tensor]], valid_pred: Tensor,
                        batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        results = []
        for i in range(len(batch_input_metas)):
            results.append(
                self._predict_by_feat_single(
                    center_preds=[x[i] for x in center_preds],
                    bbox_preds=[x[i] for x in bbox_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    valid_preds=[x[i] for x in valid_preds],
                    input_meta=batch_input_metas[i]))
        return results

    @staticmethod
    def _upsample_valid_preds(valid_pred, features):
        """Upsample valid mask predictions.

        Args:
            valid_pred (Tensor): Valid mask prediction.
            features (Tensor): Feature tensor.

        Returns:
            tuple[Tensor]: Upsampled valid masks for all feature levels.
        """
        return [
            nn.Upsample(size=x.shape[-3:],
                        mode='trilinear')(valid_pred).round().bool()
            for x in features
        ]

    @torch.no_grad()
    def _get_points(self, featmap_sizes, origin, device):
        mlvl_points = []
        tmp_voxel_size = [.16, .16, .2]
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(
                get_points(
                    n_voxels=torch.tensor(featmap_size),
                    voxel_size=torch.tensor(tmp_voxel_size) * (2**i),
                    origin=torch.tensor(origin)).reshape(3, -1).transpose(
                        0, 1).to(device))
        return mlvl_points

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3).
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, alpha ->
        # x_center, y_center, z_center, w, l, h, alpha
        shift = torch.stack(((bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
                             (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
                             (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2),
                            dim=-1).view(-1, 1, 3)
        # shift = rotation_3d_in_axis(shift, bbox_pred[:, 6], axis=2)[:, 0, :]
        shift = rotation_3d_in_axis(shift.float(), bbox_pred[:, 6].float(), axis=2)[:, 0, :]
        center = points + shift
        size = torch.stack(
            (bbox_pred[:, 0] + bbox_pred[:, 1], bbox_pred[:, 2] +
             bbox_pred[:, 3], bbox_pred[:, 4] + bbox_pred[:, 5]),
            dim=-1)
        return torch.cat((center, size, bbox_pred[:, 6:7]), dim=-1)

    # The function is directly copied from FCAF3DHead.
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        # shift = rotation_3d_in_axis(
        #     shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
        shift = rotation_3d_in_axis(
            shift.float(), -boxes[0, :, 6].float(), axis=2).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    # The function is directly copied from FCAF3DHead.
    @staticmethod
    def _get_centerness(face_distances):
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    # The function is directly copied from FCAF3DHead.
    @torch.no_grad()
    def _get_targets(self, points, gt_bboxes, gt_labels):
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification
                targets for all locations.
        """
        float_max = points[0].new_tensor(1e8)
        n_levels = len(points)
        levels = torch.cat([
            points[i].new_tensor(i).expand(len(points[i]))
            for i in range(len(points))
        ])
        points = torch.cat(points)
        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside box
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        # print(boxes)
        boxes = boxes.expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0

        # condition 2: positive points per level >= limit
        # calculate positive points per scale
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(
                torch.sum(inside_box_condition[levels == i], dim=0))
        # find best level
        n_pos_points_per_level = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_level < self.pts_assign_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(
            torch.logical_not(lower_limit_mask), dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1,
                                 lower_index)
        # keep only points with best level
        best_level = best_level.expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels

        # condition 3: limit topk points per box by centerness
        centerness = self._get_centerness(face_distances)
        centerness = torch.where(inside_box_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        centerness = torch.where(level_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(
            centerness,
            min(self.pts_center_threshold + 1, len(centerness)),
            dim=0).values[-1]
        topk_condition = centerness > top_centerness.unsqueeze(0)

        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(level_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        min_volumes, min_inds = volumes.min(dim=1)

        center_targets = centerness[torch.arange(n_points), min_inds]
        bbox_targets = boxes[torch.arange(n_points), min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)
        return center_targets, bbox_targets, cls_targets

    # Originally ImVoxelNet utilizes 2d nms as mmdetection3d didn't
    # support 3d nms. But since mmcv==1.5.2 we simply use nms3d here.
    # The function is directly copied from FCAF3DHead.
    def _single_scene_multiclass_nms(self, bboxes, scores, input_meta):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor]: Predicted bboxes, scores and labels.
        """
        n_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes.float(), class_scores.float(),
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if with_yaw:
            box_dim = 7
        else:
            box_dim = 6
            nms_bboxes = nms_bboxes[:, :box_dim]

        return nms_bboxes, nms_scores, nms_labels


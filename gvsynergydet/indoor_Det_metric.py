# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmdet.evaluation import eval_map
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import indoor_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type


@METRICS.register_module()
class GVSynergyDet_IndoorMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(GVSynergyDet_IndoorMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.ann_file = ann_file
        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = []
        pred_results = []

        for eval_ann, sinlge_pred_results in results:
            ann_infos.append(eval_ann)
            pred_results.append(sinlge_pred_results)

        # some checkpoints may not record the key "box_type_3d"
        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        ret_dict = indoor_eval(
            ann_infos,
            pred_results,
            self.iou_thr,
            self.dataset_meta['classes'],
            logger=logger,
            box_mode_3d=box_mode_3d)

        return ret_dict






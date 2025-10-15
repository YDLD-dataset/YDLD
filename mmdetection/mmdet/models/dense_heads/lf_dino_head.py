
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean)

from ..losses import QualityFocalLoss
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead

from . import DINOHead

@MODELS.register_module()
class LFDINOHead(DINOHead):

    def __init__(self,
                 use_lf_loss=False,
                 prior_file=None,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.use_lf_loss = use_lf_loss
        if use_lf_loss:
            self.prior_file = prior_file
            self.prior = None


    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """

        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                bg_class_ind = self.num_classes
                pos_inds = ((labels >= 0)
                            & (labels < bg_class_ind)).nonzero().squeeze(1)
                scores = label_weights.new_zeros(labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                scores[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
                loss_cls = self.loss_cls(
                    cls_scores, (labels, scores),
                    weight=label_weights,
                    avg_factor=cls_avg_factor)

            elif self.use_lf_loss:
                if self.prior is None:
                    self.prior = torch.tensor(np.load(self.prior_file)).to(device=cls_scores.device)
                img_w, img_h, _ = self.prior.shape

                factors = []
                for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
                    # img_h, img_w, = img_meta['img_shape']
                    factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                                   img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
                    factors.append(factor)
                factors = torch.cat(factors, 0)

                # DETR regress the relative position of boxes (cxcywh) in the image,
                # thus the learning target is normalized by the image size. So here
                # we need to re-scale them for calculating IoU loss
                prior_bbox_preds = dn_bbox_preds.reshape(-1, 4)
                prior_bboxes = (bbox_cxcywh_to_xyxy(prior_bbox_preds) * factors).long()
                num_bbox = prior_bbox_preds.shape[0]
                # psi_list = [torch.maximum(0, self.prior[prior_bboxes[i, 0]): torch.maximum(prior_bboxes[i, 2], prior_bboxes[i, 0] + 1),
                #             torch.maximum(0, prior_bboxes[i, 1]): torch.maximum(prior_bboxes[i, 3], prior_bboxes[i, 1] + 1), :].mean(dim=(0, 1))
                #             for i in range(num_bbox)]
                psi_list = [ self.prior[prior_bboxes[i, 0]: prior_bboxes[i, 2],
                                        prior_bboxes[i, 1]:prior_bboxes[i, 3],:].mean(dim=(0, 1)) for i in range(num_bbox)]

                psi = torch.stack(psi_list, dim=0)
                psi[psi.isnan()] = 0.0

                loss_cls = self.loss_cls(cls_scores, labels, label_weights, psi=psi, avg_factor=cls_avg_factor)
                #print(loss_cls)

            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)



        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

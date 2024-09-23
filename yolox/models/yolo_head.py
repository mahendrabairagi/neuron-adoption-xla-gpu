import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl


from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
def ensure_xla(tensor):
   return tensor.to(xm.xla_device())

def safe_index(tensor, mask):
    nonzero_indices = torch.nonzero(mask).squeeze(1)
    return torch.index_select(tensor, 0, nonzero_indices)

class YOLOXHead(nn.Module):

    device = xm.xla_device()

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].dtype #changed .type() to .dtype
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    # .to(xin[0].device)     
                    # .type_as(xin[0])            # EDIT, not sure if we need to also change the device
                    .to(dtype = xin[0].dtype)
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)   #EDIT HERE was .type()
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]

        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # grid = torch.stack((xv, yv), 2).view((1, 1, hsize, wsize, 2)).to(dtype=dtype, device=output.device)
            grid = torch.stack((xv, yv), 2).view((1, 1, hsize, wsize, 2)).to(dtype=dtype)

            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype=dtype, device=outputs.device)
            grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype=dtype)
            grids.append(grid)
            shape = grid.shape[:2]
            # strides.append(torch.full((*shape, 1), stride, device=outputs.device, dtype=dtype))
            strides.append(torch.full((*shape, 1), stride, dtype=dtype))

        grids = torch.cat(grids, dim=1).to(dtype=dtype) #added .to etc
        strides = torch.cat(strides, dim=1).to(dtype=dtype) #added .to etc

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        print(f"XXXXXXXXXXXXXXXXXXXX PRINTING labels = {labels}")
        print(f"XXXXXXXXXXXXXXXXXXXX PRINTING labels.dtype = {labels.dtype}")

        labels_cpu = labels.to('cpu')
        print("\n\n\n")
        print(f"XXXXXXXXXXXXXXXXXXXX PRINTING labels_cpu = {labels_cpu}")
        print(f"XXXXXXXXXXXXXXXXXXXX PRINTING labels_cpu.dtype = {labels_cpu.dtype}")
        
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]


        print(f"Step 1 - Sum along dim 2: {labels.sum(dim=2)}")
        step1 = labels.sum(dim=2) > 0
        print(f"Step 2 - Boolean mask where sum > 0: {step1}")
        nlabel = step1.sum(dim=1)
        print(f"Step 3 - Final nlabel: {nlabel}")

        
        print("\n\n\n")

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        print(f"XXXXXXXXXXXXXXXXXXXX PRINTING nlabel = {nlabel}")
        print("\n\n")
        nlabel_cpu = (labels_cpu.sum(dim=2) > 0).sum(dim=1)  # number of objects
        print(f"XXXXXXXXXXXXXXXXXXXX PRINTING nlabel_cpu = {nlabel_cpu}")

        # nlabel = nlabel_cpu
        # labels = labels_cpu
        # print(f"XXXXXXXXXXXXXXXXXXXX PRINTING NEW nlabel = {nlabel}")
        # print(f"XXXXXXXXXXXXXXXXXXXX PRINTING nlabel[0] = {nlabel[0]}")


        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        # expanded_strides = torch.cat(expanded_strides, 1).to(device=outputs.device)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX gt_classes = labels[batch_idx: num_gt, 0] = {batch_idx, num_gt, gt_classes}")
                bboxes_preds_per_image = bbox_preds[batch_idx]

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    batch_idx,
                    num_gt,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    obj_preds,
                )

                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    #expanded_strides = ensure_xla(expanded_strides)
                    #fg_mask = ensure_xla(fg_mask)
                    # Ensure tensors are on XLA device
                    x_shifts = x_shifts.to(xm.xla_device())
                    y_shifts = y_shifts.to(xm.xla_device())
                    expanded_strides = expanded_strides.to(xm.xla_device())
                    fg_mask = fg_mask.to(xm.xla_device())
                    # Use safe_index for all similar operations
                    #x_shifts = safe_index(x_shifts[0], fg_mask)
                    #y_shifts = safe_index(y_shifts[0], fg_mask)
                    #expanded_strides = safe_index(expanded_strides[0], fg_mask)
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
			x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            # obj_targets.append(obj_target.to(dtype=dtype, device=outputs.device))
            obj_targets.append(obj_target.to(dtype=dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)


        # print(f"****************PRINTING GT CLASSES AFTER DEFINITION = {gt_classes}")

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX FINISHED GET LOSS FUNCTION!!")

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        # device = xm.xla_device()
        # gt_classes = gt_classes.to(device)
        # gt_bboxes_per_image = gt_bboxes_per_image.to(device)

        # nlabel = nlabel.to(device)
        # labels = labels.to(device)
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX MODE: {mode}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX gt_classes device: {gt_classes.device}")
        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        print(f"XXXXXXXXXXXXXXXXXXXXXX Shape of fg_mask: {fg_mask.shape}, sum: {fg_mask.sum()}")
        print(f"XXXXXXXXXXXXXXXXXXXX Shape of bboxes_preds_per_image before masking: {bboxes_preds_per_image.shape}\n")
        print(f"Shape of cls_preds before masking : {cls_preds.shape}")
        # print(f"Sample content of cls_preds (first element): {cls_preds[0]}\n")
        print(f"Shape of obj_preds before masking : {obj_preds.shape}\n")
        # print(f"Sample content of obj_preds (first element): {obj_preds[0]}\n")

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        print("AAAAAAAAAAAA LINE 462")
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        print("AAAAAAAAAAAA LINE 464")
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        print("AAAAAAAAAAAA LINE 466")
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        print("AAAAAAAAAAAA LINE 468")


        print(f"\nShape of cls_preds after masking : {cls_preds_.shape}")
        # print(f"Sample content of cls_preds (first element): {cls_preds[0]}\n")
        print(f"Shape of obj_preds after masking : {obj_preds_.shape}\n")
        # print(f"Sample content of obj_preds (first element): {obj_preds[0]}\n")


        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        print(f"XXXXXXXXXXXXXX gt_bboxes_per_image device: {gt_bboxes_per_image.device}")
        print(f"XXXXXXXXXXXXXX bboxes_preds_per_image device: {bboxes_preds_per_image.device}")

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        print("AAAAAAAAAAAA LINE 476")

        #########################################
        ################ EDIT ###################
        #########################################
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX DTYPE = {gt_classes.dtype}")
        # gt_classes = gt_classes.to(torch.int64)
        # gt_classes = gt_classes.to('cpu')
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX GT_CLASSES = {gt_classes}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX GT_CLASSES.norm() = {gt_classes.norm()}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX Class values norm: {gt_classes.norm().item()}, max: {gt_classes.max().item()}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX GT_CLASSES.max() = {gt_classes.max()}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX GT_CLASSES_SHAPE = {gt_classes.shape}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX NUM_CLASSES = {self.num_classes}")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXX DTYPE = {gt_classes.dtype}")

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        print(f"XXXXXXXXXXXXXXXXX gt_cls_per_image = {gt_cls_per_image} , SIZE = {gt_cls_per_image.size()}")
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        print("AAAAAAAAAAAA LINE 506")


        with torch.cuda.amp.autocast(enabled=False):
            print(f"Shape of cls_preds_ before multiplication: {cls_preds_.shape}")
            print(f"Sample content of cls_preds_ (first element): {cls_preds_[0]}\n")

            print(f"Shape of obj_preds_ before multiplication: {obj_preds_.shape}")
            print(f"Sample content of obj_preds_ (first element): {obj_preds_[0]}\n")

            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()

            print(f"Shape of cls_preds_ after multiplication and before unsqueeze and repeat: {cls_preds_.shape}")
            print(f"Sample content of cls_preds_ (first element): {cls_preds_[0]}\n")

            print(f"Shape of gt_cls_per_image after multiplication and beforebefore unsqueeze and repeat: {gt_cls_per_image.shape}")
            print(f"Sample content of gt_cls_per_image (first element): {gt_cls_per_image[0]}\n")

            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        print("AAAAAAAAAAAA LINE 519")

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        print("AAAAAAAAAAAA LINE 527")

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].dtype)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).to(dtype=xin[0].dtype, device=xin[0].device)
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")

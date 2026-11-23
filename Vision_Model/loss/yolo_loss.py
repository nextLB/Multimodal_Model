
import torch
import torch.nn.functional as F


def _iter_loss_branches(criterion, preds):
    one2many = getattr(criterion, "one2many", None)
    one2one = getattr(criterion, "one2one", None)

    if (
        one2many is not None
        and isinstance(preds, dict)
        and isinstance(preds.get("one2many"), dict)
    ):
        yield one2many, preds["one2many"], float(getattr(criterion, "o2m", 1.0))
        if one2one is not None and isinstance(preds.get("one2one"), dict):
            yield one2one, preds["one2one"], float(getattr(criterion, "o2o", 0.0))
        return

    core = one2many if one2many is not None else criterion
    yield core, preds, 1.0


class YOLO26DetectionLoss_Ultralytics:
    """
    YOLO26 目标检测损失计算类。

    使用官方 SegmentLoss26 完成：
        1. Task Aligned Target Assignment
        2. Box Loss
        3. Classification Loss
        4. DFL Loss

    同时返回目标分配信息，供 Mask Loss 共享。
    """

    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, preds, batch):
        total_det_loss = 0.0
        branches = []
        loss_items = {
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "dfl_loss": 0.0,
        }
        names = ("box_loss", "cls_loss", "dfl_loss")

        for core, branch_preds, weight in _iter_loss_branches(self.criterion, preds):
            parsed = core.parse_output(branch_preds)

            assignment, det_components, _ = (
                core.get_assigned_targets_and_loss(parsed, batch)
            )

            batch_size = parsed["boxes"].shape[0]
            total_det_loss = (
                total_det_loss + det_components.sum() * batch_size * weight
            )

            branches.append({
                "criterion": core,
                "preds": parsed,
                "assignment": assignment,
                "batch_size": batch_size,
                "weight": weight,
            })

            for name, component in zip(names, det_components):
                loss_items[name] = loss_items[name] + component.detach()

        context = {"branches": branches}

        return total_det_loss, context, loss_items


class YOLO26MaskLoss_Ultralytics:
    """
    YOLO26 实例分割损失计算类。

    使用官方 SegmentLoss26 的：
        calculate_segmentation_loss()

    计算实例级 Mask BCE Loss。
    """

    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, batch, context):
        total_seg_loss = 0.0

        for branch in context["branches"]:
            core = branch["criterion"]
            preds = branch["preds"]
            fg_mask, target_gt_idx, target_bboxes, _, _ = branch["assignment"]

            pred_masks = (
                preds["mask_coefficient"]
                .permute(0, 2, 1)
                .contiguous()
            )

            proto = preds["proto"]
            if isinstance(proto, tuple):
                proto = proto[0]

            masks = batch["masks"].to(
                device=core.device
            ).float()

            if not fg_mask.any():
                seg_loss = (proto * 0).sum() + (pred_masks * 0).sum()
            else:
                if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
                    proto = F.interpolate(
                        proto,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                imgsz = torch.tensor(
                    preds["feats"][0].shape[2:],
                    device=core.device,
                    dtype=pred_masks.dtype,
                ) * core.stride[0]

                seg_loss = core.calculate_segmentation_loss(
                    fg_mask=fg_mask,
                    masks=masks,
                    target_gt_idx=target_gt_idx,
                    target_bboxes=target_bboxes,
                    batch_idx=batch["batch_idx"].view(-1, 1),
                    proto=proto,
                    pred_masks=pred_masks,
                    imgsz=imgsz,
                )

            seg_loss = (
                seg_loss
                * core.hyp.box
                * branch["batch_size"]
                * branch["weight"]
            )
            total_seg_loss = total_seg_loss + seg_loss

        return total_seg_loss, {
            "mask_loss": total_seg_loss.detach()
        }




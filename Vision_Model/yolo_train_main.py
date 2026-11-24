# yolo_train_main.py

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.nms import non_max_suppression, box_iou
from ultralytics.utils.metrics import ap_per_class

from datasets.dataset import (
    PascalVoc2012,
    HqImprove1022,
    UnifiedDetectionSegmentationDataset,
    detection_collate_fn,
)
from loss.yolo_loss import (
    YOLO26DetectionLoss_Ultralytics,
    YOLO26MaskLoss_Ultralytics,
)


# ============================================================
# 配置
# ============================================================

PRETRAINED_WEIGHTS = r"D:\LB\results\pre_models\yolo26m-seg.pt" # n / s / m / l / x
SAVE_DIR = r"D:\LB\results\runs/train_yolo26n_seg_unified"
RUN_NAME = "exp01"

VOC_ROOT = r"D:\LB\Datasets\PASCAL_VOC2012\VOCdevkit\VOC2012"
HQ_ROOT = r"D:\LB\Datasets\Hq_Datasets\Improve_Dataset_1022"
IMAGE_SIZE = (640, 640)

EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
CLOSE_MOSAIC = 10

BOX_GAIN = 7.5
CLS_GAIN = 0.5
DFL_GAIN = 1.5
SEG_GAIN = 1.0

VAL_INTERVAL = 1
VAL_BATCH_SIZE = 8
VAL_CONF = 0.001
VAL_IOU = 0.6
VAL_MAX_DET = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

LOG_INTERVAL = 10
SAVE_PERIOD = 10

LOG_ROOT = r"D:\LB\results\runs/logs"
LOG_NAME = "train"


# ============================================================
# 日志
# ============================================================

logger = logging.getLogger("yolo_train")


class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty() if hasattr(self.stream, "isatty") else False


def setup_logging():
    root = logging.getLogger()
    if getattr(root, "_yolo_train_ready", False):
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(LOG_ROOT) / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{LOG_NAME}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = sys.stdout
    stream_handler = logging.StreamHandler(console)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(logging.INFO)

    tee_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(console, tee_file)
    sys.stderr = TeeStream(sys.stderr, tee_file)

    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.handlers.clear()
    ultralytics_logger.propagate = True

    root._yolo_train_ready = True
    return log_path


# ============================================================
# 训练辅助
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _format_bytes(num_bytes):
    gib = float(num_bytes) / (1024 ** 3)
    return f"{gib:.2f}GiB"


def get_resource_usage(device=DEVICE):
    usage = {}
    process = psutil.Process(os.getpid()) if psutil is not None else None

    if process is not None:
        usage["proc_cpu"] = process.cpu_percent(interval=None)
        mem_info = process.memory_info()
        usage["proc_mem"] = mem_info.rss
        usage["sys_cpu"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        usage["sys_mem_used"] = vm.used
        usage["sys_mem_total"] = vm.total
    else:
        try:
            load = os.getloadavg()
            usage["loadavg"] = load[0]
        except Exception:
            pass

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            dev = torch.device(device)
            dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
            usage["gpu_name"] = torch.cuda.get_device_name(dev_idx)
            usage["gpu_mem_alloc"] = torch.cuda.memory_allocated(dev_idx)
            usage["gpu_mem_reserved"] = torch.cuda.memory_reserved(dev_idx)
            if hasattr(torch.cuda, "max_memory_allocated"):
                usage["gpu_mem_max_alloc"] = torch.cuda.max_memory_allocated(dev_idx)
            total = None
            if hasattr(torch.cuda, "get_device_properties"):
                total = torch.cuda.get_device_properties(dev_idx).total_memory
            if total is not None:
                usage["gpu_mem_total"] = total
            util = None
            if hasattr(torch.cuda, "utilization"):
                try:
                    util = torch.cuda.utilization(dev_idx)
                except Exception:
                    util = None
            if util is not None:
                usage["gpu_util"] = util
        except Exception as exc:
            usage["gpu_error"] = str(exc)

    return usage


def format_resource_usage(usage, device=DEVICE):
    parts = [f"device={device}"]
    if "proc_cpu" in usage:
        parts.append(f"proc_cpu={usage['proc_cpu']:.1f}%")
    if "sys_cpu" in usage:
        parts.append(f"sys_cpu={usage['sys_cpu']:.1f}%")
    if "proc_mem" in usage:
        parts.append(f"proc_mem={_format_bytes(usage['proc_mem'])}")
    if "sys_mem_used" in usage and "sys_mem_total" in usage:
        parts.append(
            f"sys_mem={_format_bytes(usage['sys_mem_used'])}/{_format_bytes(usage['sys_mem_total'])}"
        )
    if "loadavg" in usage:
        parts.append(f"loadavg1={usage['loadavg']:.2f}")
    if "gpu_name" in usage:
        parts.append(f"gpu={usage['gpu_name']}")
    if "gpu_util" in usage:
        parts.append(f"gpu_util={usage['gpu_util']}%")
    if "gpu_mem_alloc" in usage:
        gpu_mem = f"gpu_mem={_format_bytes(usage['gpu_mem_alloc'])}"
        if "gpu_mem_reserved" in usage:
            gpu_mem += f"/reserved {_format_bytes(usage['gpu_mem_reserved'])}"
        if "gpu_mem_total" in usage:
            gpu_mem += f"/total {_format_bytes(usage['gpu_mem_total'])}"
        parts.append(gpu_mem)
    if "gpu_mem_max_alloc" in usage:
        parts.append(f"gpu_max_alloc={_format_bytes(usage['gpu_mem_max_alloc'])}")
    if "gpu_error" in usage:
        parts.append(f"gpu_error={usage['gpu_error']}")
    if len(parts) == 1:
        parts.append("resource=unavailable")
    return " | ".join(parts)


def log_resource_usage(prefix="[Resource]", device=DEVICE):
    usage = get_resource_usage(device=device)
    logger.info("%s %s", prefix, format_resource_usage(usage, device=device))
    return usage


def move_batch_to_device(images, targets, device):
    if isinstance(images, torch.Tensor):
        images = images.to(device, non_blocking=True)
    else:
        images = [img.to(device, non_blocking=True) for img in images]

    for t in targets:
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                t[k] = v.to(device, non_blocking=True)

    return images, targets


def build_ultralytics_batch(images, targets):
    batch_size = images.shape[0]
    img_h, img_w = images.shape[-2], images.shape[-1]
    batch_idx_list, cls_list, bboxes_list = [], [], []

    masks = torch.zeros(
        (batch_size, img_h, img_w),
        dtype=torch.float32,
        device=images.device,
    )

    for sample_idx, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]

        if boxes.shape[0]:
            cx = (boxes[:, 0] + boxes[:, 2]) / (2.0 * img_w)
            cy = (boxes[:, 1] + boxes[:, 3]) / (2.0 * img_h)
            bw = (boxes[:, 2] - boxes[:, 0]) / img_w
            bh = (boxes[:, 3] - boxes[:, 1]) / img_h
            bboxes_list.append(torch.stack((cx, cy, bw, bh), dim=1))
            cls_list.append(labels.long())
            batch_idx_list.append(
                torch.full(
                    (boxes.shape[0],),
                    sample_idx,
                    dtype=torch.long,
                    device=images.device,
                )
            )

        instance_boxes = target.get("instance_boxes")
        instance_masks = target.get("instance_masks")
        if (
            instance_boxes is not None
            and instance_masks is not None
            and instance_boxes.shape[0]
            and instance_masks.shape[0]
        ):
            for instance_idx in range(instance_masks.shape[0]):
                matched = (
                    boxes == instance_boxes[instance_idx]
                ).all(dim=1).nonzero(as_tuple=False)
                if matched.numel() == 0:
                    continue
                detection_idx = int(matched[0].item())
                masks[sample_idx][
                    instance_masks[instance_idx].to(
                        device=images.device,
                        dtype=torch.bool,
                    )
                ] = detection_idx + 1

    if bboxes_list:
        batch_idx = torch.cat(batch_idx_list)
        cls = torch.cat(cls_list)
        bboxes = torch.cat(bboxes_list)
    else:
        batch_idx = torch.zeros(0, dtype=torch.long, device=images.device)
        cls = torch.zeros(0, dtype=torch.long, device=images.device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=images.device)

    return {
        "img": images,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "masks": masks,
    }


def rebuild_head_for_new_nc(yolo, num_classes, verbose=True):
    old_net = yolo.model
    old_nc = old_net.model[-1].nc

    if verbose:
        logger.info(f"检测头 nc={old_nc} → {num_classes}，重建 Segmentation 头")

    new_net = SegmentationModel(
        cfg=dict(old_net.yaml),
        nc=num_classes,
        verbose=False,
    )

    old_sd = old_net.state_dict()
    new_sd = new_net.state_dict()
    compatible = {
        k: v for k, v in old_sd.items()
        if k in new_sd and v.shape == new_sd[k].shape
    }
    missing, unexpected = new_net.load_state_dict(compatible, strict=False)

    if verbose:
        logger.info(
            f"  成功加载 {len(compatible)} 个参数；"
            f"随机初始化 {len(missing)} 个（分类头通道数变化）"
        )
        if unexpected:
            logger.info(f"  意外键（应为空）: {unexpected[:5]} ...")

    return new_net


def build_validation_loader(expected_classes):
    voc_val = PascalVoc2012(
        root=VOC_ROOT,
        split="val",
        task="both",
        image_size=IMAGE_SIZE,
    )
    hq_val = HqImprove1022(
        root=HQ_ROOT,
        split="val",
        task="both",
        image_size=IMAGE_SIZE,
    )
    val_dataset = UnifiedDetectionSegmentationDataset([voc_val, hq_val])

    if val_dataset.num_detection_classes != expected_classes:
        raise ValueError(
            f"Validation classes mismatch: "
            f"{val_dataset.num_detection_classes} != {expected_classes}"
        )
    return DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================
# 验证评估
# ============================================================

def match_predictions(pred_classes, true_classes, iou, iouv):
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if true_classes.shape[0] == 0 or pred_classes.shape[0] == 0:
        return correct

    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    thresholds = iouv.cpu().numpy()
    matched = np.zeros((iou.shape[0], len(thresholds)), dtype=bool)

    for det_idx in np.flatnonzero((iou >= thresholds.min()).any(0)):
        available = np.where(matched, 0, iou[:, det_idx, None])
        gt_idx = available.argmax(0)
        correct[det_idx] = available[gt_idx, range(len(thresholds))] >= thresholds
        matched[gt_idx, range(len(thresholds))] |= correct[det_idx]

    return correct


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_names):
    was_training = model.training
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images)
        detections = non_max_suppression(
            preds[0],
            conf_thres=VAL_CONF,
            iou_thres=VAL_IOU,
            nc=num_classes,
            max_det=VAL_MAX_DET,
        )

        for sample_idx, det in enumerate(detections):
            gt_labels = targets[sample_idx]["labels"].to(device)
            target_cls_list.append(gt_labels.cpu().numpy())

            if det is None or det.shape[0] == 0:
                continue

            gt_boxes = targets[sample_idx]["boxes"].to(device)
            if gt_boxes.shape[0]:
                iou = box_iou(gt_boxes, det[:, :4])
            else:
                iou = torch.zeros((0, det.shape[0]), device=device)

            tp_list.append(match_predictions(det[:, 5], gt_labels, iou, iouv))
            conf_list.append(det[:, 4].cpu().numpy())
            pred_cls_list.append(det[:, 5].cpu().numpy())

    model.train(was_training)

    empty = {
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "ap50_by_class": {},
    }
    if not target_cls_list:
        return empty

    target_cls = np.concatenate(target_cls_list)
    if target_cls.size == 0:
        return empty

    tp = (
        np.concatenate(tp_list)
        if tp_list else np.zeros((0, 10), dtype=bool)
    )
    conf = (
        np.concatenate(conf_list)
        if conf_list else np.zeros((0,), dtype=np.float32)
    )
    pred_cls = (
        np.concatenate(pred_cls_list)
        if pred_cls_list else np.zeros((0,), dtype=np.int64)
    )

    stats = ap_per_class(tp, conf, pred_cls, target_cls)
    ap = stats[5]
    ap_classes = stats[6]

    return {
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "precision": float(stats[2].mean()),
        "recall": float(stats[3].mean()),
        "f1": float(stats[4].mean()),
        "ap50_by_class": {
            int(cls): float(ap[i, 0])
            for i, cls in enumerate(ap_classes)
        },
    }


def print_validation(epoch, metrics, class_names):
    logger.info(
        f"\n===== Validation Epoch {epoch}/{EPOCHS} =====\n"
        f"  mAP@0.5:      {metrics['map50']:.4f}\n"
        f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}\n"
        f"  Precision:    {metrics['precision']:.4f}\n"
        f"  Recall:       {metrics['recall']:.4f}\n"
        f"  F1:           {metrics['f1']:.4f}"
    )
    if metrics["ap50_by_class"]:
        items = "  ".join(
            f"{class_names[cls]}={value:.3f}"
            for cls, value in metrics["ap50_by_class"].items()
        )
        logger.info(f"  AP@0.5: {items}")


# ============================================================
# 主流程
# ============================================================

def main():
    log_path = setup_logging()
    logger.info("日志文件: %s", log_path)
    logger.info(
        "配置: epochs=%d, batch=%d, image_size=%s, device=%s, "
        "lr0=%.5f, lrf=%.5f, warmup_epochs=%d, val_interval=%d",
        EPOCHS, BATCH_SIZE, IMAGE_SIZE, DEVICE,
        LR0, LRF, WARMUP_EPOCHS, VAL_INTERVAL,
    )
    if psutil is None:
        logger.info("psutil 未安装，CPU/内存监控将受限；可安装 psutil 获得完整信息")
    log_resource_usage("[Resource][startup]", device=DEVICE)
    seed_everything(SEED)

    voc = PascalVoc2012(
        root=VOC_ROOT,
        split="train",
        task="both",
        image_size=IMAGE_SIZE,
    )
    hq = HqImprove1022(
        root=HQ_ROOT,
        split="train",
        task="both",
        image_size=IMAGE_SIZE,
    )
    train_dataset = UnifiedDetectionSegmentationDataset([voc, hq])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    num_classes = train_dataset.num_detection_classes
    logger.info(f"训练集总样本数: {len(train_dataset)}")
    logger.info(f"检测类别数: {num_classes}")
    logger.info(f"语义分割类别数（含背景）: {train_dataset.num_segmentation_classes}")

    val_loader = build_validation_loader(num_classes)
    logger.info(f"验证集总样本数: {len(val_loader.dataset)}")

    yolo = YOLO(PRETRAINED_WEIGHTS)
    net = yolo.model
    orig_args_dict = (
        dict(net.args) if isinstance(net.args, dict)
        else dict(DEFAULT_CFG_DICT)
    )

    if net.model[-1].nc != num_classes:
        net = rebuild_head_for_new_nc(yolo, num_classes, verbose=True)
    else:
        logger.info(f"检测头 nc 已匹配 ({num_classes})，跳过重建")

    args = DEFAULT_CFG_DICT.copy()
    args.update(orig_args_dict)
    args.update({
        "nc": num_classes,
        "box": BOX_GAIN,
        "cls": CLS_GAIN,
        "dfl": DFL_GAIN,
        "overlap_mask": True,
        "mask_ratio": 4,
    })
    net.args = IterableSimpleNamespace(**args)
    net.nc = num_classes
    net.names = {
        i: name for i, name in enumerate(train_dataset.global_class_names)
    }

    net = net.to(DEVICE)
    net.train()

    criterion = net.init_criterion()
    core_criteria = []
    if hasattr(criterion, "one2many"):
        core_criteria.append(criterion.one2many)
    if hasattr(criterion, "one2one"):
        core_criteria.append(criterion.one2one)
    if not core_criteria:
        core_criteria.append(criterion)

    for c in core_criteria:
        if not hasattr(c, "hyp"):
            from types import SimpleNamespace
            c.hyp = SimpleNamespace(
                box=BOX_GAIN,
                cls=CLS_GAIN,
                dfl=DFL_GAIN,
            )
        else:
            c.hyp.box = BOX_GAIN
            c.hyp.cls = CLS_GAIN
            c.hyp.dfl = DFL_GAIN

    det_loss_fn = YOLO26DetectionLoss_Ultralytics(criterion)
    mask_loss_fn = YOLO26MaskLoss_Ultralytics(criterion)

    net.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]
    num_param_tensors = len(params)
    num_param_elems = sum(p.numel() for p in params)
    logger.info(
        f"可训练参数张量数: {num_param_tensors}, "
        f"可训练参数总量: {num_param_elems:,}"
    )
    if not params:
        raise RuntimeError("模型没有可训练参数")

    optimizer = torch.optim.SGD(
        params,
        lr=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return LRF + (1 - LRF) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(SAVE_DIR) / RUN_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    val_metrics = {}
    for epoch in range(EPOCHS):

        if hasattr(net, "args") and CLOSE_MOSAIC > 0:
            net.args.close_mosaic = (epoch >= EPOCHS - CLOSE_MOSAIC)

        net.train()
        epoch_det = 0.0
        epoch_mask = 0.0
        epoch_total = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = move_batch_to_device(images, targets, DEVICE)
            batch = build_ultralytics_batch(images, targets)

            preds_raw = net(images)
            det_loss, context, _ = det_loss_fn(preds_raw, batch)
            mask_loss, _ = mask_loss_fn(batch, context)
            loss = det_loss + SEG_GAIN * mask_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()
            scheduler.step()

            if hasattr(criterion, "update"):
                criterion.update()

            epoch_det += det_loss.item()
            epoch_mask += mask_loss.item()
            epoch_total += loss.item()
            num_batches += 1

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Iter [{batch_idx+1}/{len(train_loader)}] "
                    f"LR: {lr:.6f} | "
                    f"Det: {det_loss.item():.4f} | "
                    f"Mask: {mask_loss.item():.4f} | "
                    f"Total: {loss.item():.4f}"
                )
                log_resource_usage(
                    f"[Resource][epoch {epoch+1} iter {batch_idx+1}]",
                    device=DEVICE,
                )

            global_step += 1

        avg_det = epoch_det / max(1, num_batches)
        avg_mask = epoch_mask / max(1, num_batches)
        avg_total = epoch_total / max(1, num_batches)

        logger.info(
            f"\n===== Epoch {epoch+1}/{EPOCHS} 完成 =====\n"
            f"  Det Loss:  {avg_det:.4f}\n"
            f"  Mask Loss: {avg_mask:.4f}\n"
            f"  Total:     {avg_total:.4f}\n"
        )
        log_resource_usage(f"[Resource][epoch {epoch+1} done]", device=DEVICE)

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_metrics = evaluate(
                net, val_loader, DEVICE, num_classes, net.names
            )
            print_validation(epoch + 1, val_metrics, net.names)

        if (epoch + 1) % SAVE_PERIOD == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = save_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "det_loss": avg_det,
                    "mask_loss": avg_mask,
                    "total_loss": avg_total,
                    "val_metrics": val_metrics,
                    "num_detection_classes": num_classes,
                    "global_class_names": train_dataset.global_class_names,
                },
                ckpt_path,
            )
            logger.info(f"权重已保存: {ckpt_path}")

    final_path = save_dir / "final.pt"
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "num_detection_classes": num_classes,
            "num_segmentation_classes": train_dataset.num_segmentation_classes,
            "global_class_names": train_dataset.global_class_names,
            "val_metrics": val_metrics,
        },
        final_path,
    )
    logger.info(f"最终权重已保存: {final_path}")


if __name__ == "__main__":
    main()

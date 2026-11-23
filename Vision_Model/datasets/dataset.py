"""
dataset.py

VOC2012 + HqImprove1022 统一检测 / 实例分割数据集加载器。

主要类：
    1. PascalVoc2012
    2. HqImprove1022
    3. UnifiedDetectionSegmentationDataset
    4. detection_collate_fn

统一返回：
    image:
        FloatTensor [3, H, W], RGB, range [0, 1]

    target:
        boxes: FloatTensor [N, 4]，所有检测目标的 xyxy 像素坐标
        labels: LongTensor [N]，所有检测目标的类别 ID，前景从 0 开始
        instance_boxes: FloatTensor [M, 4]，具有有效实例 Mask 的目标框
        instance_labels: LongTensor [M]，对应实例目标的类别 ID
        instance_masks: FloatTensor/BoolTensor [M, H, W]，每个实例独立的二值 Mask
        mask: LongTensor [H, W]，语义分割 Mask，0=背景，1..K=前景，255=ignore
        has_semantic_mask / has_instance_masks / has_instance_annotation: bool
        source_dataset: str，数据来源名称

类别映射：
    VOC2012: detection 0..19, semantic 1..20
    HqImprove1022: detection 0..5, semantic 1..6
    Unified: VOC detection 0..19, Hq detection 20..25
             semantic background=0, VOC=1..20, Hq=21..26, ignore=255
"""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
VOC_CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def _image_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def _empty_boxes():
    return torch.zeros((0, 4), dtype=torch.float32)


def _empty_labels():
    return torch.zeros((0,), dtype=torch.long)


def _empty_instance_masks(height, width):
    return torch.zeros((0, height, width), dtype=torch.bool)


def _resize_image_and_target(image, target, image_size):
    if image_size is None:
        return image, target

    new_h, new_w = map(int, image_size)
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid image_size: {image_size}")

    old_w, old_h = image.size
    if old_w <= 0 or old_h <= 0:
        raise ValueError(f"Invalid original image size: {image.size}")

    scale_x, scale_y = new_w / old_w, new_h / old_h
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    for key in ("boxes", "instance_boxes"):
        if key not in target:
            continue
        boxes = target[key].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        target[key] = boxes

    if "mask" in target:
        mask_image = Image.fromarray(target["mask"].cpu().numpy().astype(np.uint8))
        mask_image = mask_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        target["mask"] = torch.from_numpy(
            np.asarray(mask_image, dtype=np.uint8).copy()
        ).long()

    if "instance_masks" in target:
        resized = []
        for instance_mask in target["instance_masks"]:
            mask_image = Image.fromarray(instance_mask.cpu().numpy().astype(np.uint8))
            mask_image = mask_image.resize((new_w, new_h), Image.Resampling.NEAREST)
            resized.append(torch.from_numpy(np.asarray(mask_image, dtype=np.uint8).copy() > 0))
        target["instance_masks"] = (
            torch.stack(resized, dim=0)
            if resized else _empty_instance_masks(new_h, new_w)
        )

    return image, target


def _validate_boxes_and_labels(target, source_name):
    boxes, labels = target["boxes"], target["labels"]
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(
            f"{source_name}: boxes must have shape [N,4], got {tuple(boxes.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"{source_name}: labels must have shape [N], got {tuple(labels.shape)}"
        )
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{source_name}: boxes/labels count mismatch: "
            f"{boxes.shape[0]} vs {labels.shape[0]}"
        )


class PascalVoc2012(Dataset):
    def __init__(
        self,
        root,
        split="train",
        task="detection",
        image_size=None,
        include_difficult=False,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if task not in ("detection", "segmentation", "both"):
            raise ValueError("task must be 'detection', 'segmentation', or 'both'")

        root = Path(root)
        if (root / "VOCdevkit" / "VOC2012").is_dir():
            root = root / "VOCdevkit" / "VOC2012"
        elif root.name != "VOC2012" and (root / "VOC2012").is_dir():
            root = root / "VOC2012"
        if not root.is_dir():
            raise FileNotFoundError(f"VOC2012 directory not found: {root}")

        self.root = root
        self.split = split
        self.task = task
        self.image_size = image_size
        self.include_difficult = include_difficult
        self.class_names = list(VOC_CLASSES)
        self.num_detection_classes = len(VOC_CLASSES)
        self.num_segmentation_classes = len(VOC_CLASSES) + 1

        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        self.semantic_mask_dir = self.root / "SegmentationClass"
        self.instance_mask_dir = self.root / "SegmentationObject"

        self.detection_ids = self._read_split_ids(
            self.root / "ImageSets" / "Main" / f"{split}.txt"
        )
        segmentation_split_path = (
            self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        )
        self.segmentation_ids = (
            self._read_split_ids(segmentation_split_path)
            if segmentation_split_path.exists() else []
        )
        self.segmentation_id_set = set(self.segmentation_ids)
        self.ids = list(
            self.detection_ids
            if task in ("detection", "both") else self.segmentation_ids
        )
        if not self.ids:
            raise RuntimeError(
                f"No samples found for split={split}, task={task}. "
                "Check VOC2012 installation and ImageSets files."
            )

        print(
            f"[PascalVoc2012] split={split}, task={task}, "
            f"samples={len(self.ids)}, "
            f"segmentation_split_samples={len(self.segmentation_ids)}"
        )

    @staticmethod
    def _read_split_ids(txt_path):
        if not txt_path.exists():
            raise FileNotFoundError(f"Split file not found: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            return [
                line.strip().split()[0]
                for line in f if line.strip()
            ]

    def __len__(self):
        return len(self.ids)

    def _load_detection_and_instance_targets(self, image_id, height, width):
        xml_path = self.annotation_dir / f"{image_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"VOC annotation missing: {xml_path}")
        xml_root = ET.parse(xml_path).getroot()

        instance_mask_path = self.instance_mask_dir / f"{image_id}.png"
        has_instance_annotation = (
            image_id in self.segmentation_id_set
            and instance_mask_path.exists()
        )
        object_mask = None
        if has_instance_annotation:
            object_mask = np.asarray(
                Image.open(instance_mask_path), dtype=np.uint8
            ).copy()
            if object_mask.ndim != 2:
                raise RuntimeError(
                    "Expected 2D VOC SegmentationObject mask, "
                    f"got shape={object_mask.shape}: {instance_mask_path}"
                )
            if object_mask.shape != (height, width):
                raise ValueError(
                    f"VOC image/instance mask size mismatch: "
                    f"image={(height, width)}, mask={object_mask.shape}, "
                    f"id={image_id}"
                )

        boxes, labels = [], []
        instance_boxes, instance_labels, instance_masks = [], [], []

        for original_object_index, obj in enumerate(
            xml_root.findall("object"), start=1
        ):
            name = obj.findtext("name")
            difficult = int(obj.findtext("difficult", default="0"))
            if name not in VOC_CLASS_TO_ID:
                continue
            class_id = VOC_CLASS_TO_ID[name]
            if difficult == 1 and not self.include_difficult:
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                xmin = float(bndbox.findtext("xmin"))
                ymin = float(bndbox.findtext("ymin"))
                xmax = float(bndbox.findtext("xmax"))
                ymax = float(bndbox.findtext("ymax"))
            except (TypeError, ValueError):
                continue

            x1 = np.clip(xmin - 1.0, 0, width)
            y1 = np.clip(ymin - 1.0, 0, height)
            x2 = np.clip(xmax, 0, width)
            y2 = np.clip(ymax, 0, height)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)

            if not has_instance_annotation:
                continue
            binary_mask = object_mask == original_object_index
            if not np.any(binary_mask):
                continue
            instance_boxes.append([x1, y1, x2, y2])
            instance_labels.append(class_id)
            instance_masks.append(torch.from_numpy(binary_mask.copy()).bool())

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "instance_boxes": torch.tensor(instance_boxes, dtype=torch.float32).reshape(-1, 4),
            "instance_labels": torch.tensor(instance_labels, dtype=torch.long),
            "instance_masks": (
                torch.stack(instance_masks, dim=0)
                if instance_masks else _empty_instance_masks(height, width)
            ),
            "has_instance_annotation": bool(has_instance_annotation),
            "has_instance_masks": bool(len(instance_masks)),
        }

    def _load_semantic_mask(self, image_id, height, width):
        mask_path = self.semantic_mask_dir / f"{image_id}.png"
        has_semantic_mask = (
            image_id in self.segmentation_id_set and mask_path.exists()
        )
        if not has_semantic_mask:
            return torch.full((height, width), 255, dtype=torch.long), False

        mask_array = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()
        if mask_array.ndim != 2:
            raise RuntimeError(
                f"Expected 2D VOC semantic mask, got {mask_array.shape}: {mask_path}"
            )
        if mask_array.shape != (height, width):
            raise ValueError(
                f"VOC image/semantic mask size mismatch: "
                f"image={(height, width)}, mask={mask_array.shape}, id={image_id}"
            )
        unexpected = set(np.unique(mask_array).tolist()) - (set(range(21)) | {255})
        if unexpected:
            raise ValueError(
                f"Unexpected VOC semantic mask values: "
                f"{sorted(unexpected)} in {mask_path}"
            )
        return torch.from_numpy(mask_array).long(), True

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"VOC image missing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target = {
            "boxes": _empty_boxes(),
            "labels": _empty_labels(),
            "instance_boxes": _empty_boxes(),
            "instance_labels": _empty_labels(),
            "instance_masks": _empty_instance_masks(height, width),
            "mask": torch.full((height, width), 255, dtype=torch.long),
            "has_semantic_mask": False,
            "has_instance_annotation": False,
            "has_instance_masks": False,
        }

        if self.task in ("detection", "both", "segmentation"):
            target.update(
                self._load_detection_and_instance_targets(image_id, height, width)
            )

        if self.task in ("segmentation", "both"):
            target["mask"], target["has_semantic_mask"] = self._load_semantic_mask(
                image_id, height, width
            )

        _validate_boxes_and_labels(target, "PascalVoc2012")
        image, target = _resize_image_and_target(image, target, self.image_size)
        target["image_id"] = image_id
        return _image_to_tensor(image), target


class HqImprove1022(Dataset):
    def __init__(
        self,
        root,
        split="train",
        task="both",
        image_size=None,
        validate_mask=True,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if task not in ("detection", "segmentation", "both"):
            raise ValueError("task must be 'detection', 'segmentation', or 'both'")

        self.root = Path(root)
        self.split = split
        self.task = task
        self.image_size = image_size
        self.validate_mask = validate_mask
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        yaml_path = self.root / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        names = config.get("names", {})
        if isinstance(names, list):
            self.class_names = [str(x) for x in names]
        elif isinstance(names, dict):
            keys = sorted(int(k) for k in names.keys())
            self.class_names = [
                str(names.get(k, names.get(str(k)))) for k in keys
            ]
        else:
            raise ValueError("data.yaml 'names' must be a list or dict")

        self.nc = int(config.get("nc", len(self.class_names)))
        self.num_detection_classes = self.nc
        self.num_segmentation_classes = self.nc + 1
        if len(self.class_names) != self.nc:
            raise ValueError(
                f"data.yaml nc={self.nc}, "
                f"but names has {len(self.class_names)} entries"
            )

        self.split_dir = self.root / split
        self.raw_dir = self.split_dir / "raw"
        self.label_dir = self.split_dir / "label"
        self.mask_dir = self.split_dir / "mask"
        for directory in (self.raw_dir, self.label_dir, self.mask_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Dataset directory missing: {directory}")

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        raw_files = [
            p for p in self.raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        def natural_key(path):
            return [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", path.name)
            ]

        raw_files.sort(key=natural_key)
        self.samples = []
        for raw_path in raw_files:
            match = re.search(r"images_(\d+)", raw_path.stem)
            if match is None:
                raise ValueError(
                    f"Cannot extract sample ID from {raw_path.name}"
                )
            image_id = match.group(1)
            label_path = self.label_dir / f"labels_{image_id}.txt"
            mask_path = self.mask_dir / f"masks_{image_id}.png"
            if task in ("detection", "both") and not label_path.exists():
                raise FileNotFoundError(f"Label missing: {label_path}")
            if task in ("segmentation", "both") and not mask_path.exists():
                raise FileNotFoundError(f"Mask missing: {mask_path}")
            self.samples.append({
                "id": image_id,
                "raw": raw_path,
                "label": label_path,
                "mask": mask_path,
            })

        if not self.samples:
            raise RuntimeError(f"No images found in {self.raw_dir}")

        print(
            f"[HqImprove1022] split={split}, task={task}, "
            f"samples={len(self.samples)}, nc={self.nc}"
        )
        print(f"[HqImprove1022] classes={self.class_names}")

    def __len__(self):
        return len(self.samples)

    def _load_yolo_target(self, label_path, width, height):
        boxes, labels = [], []
        with open(label_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                values = line.split()
                if len(values) != 5:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: expected exactly 5 values: "
                        "class xc yc w h"
                    )
                try:
                    class_value = float(values[0])
                    class_id = int(class_value)
                    xc, yc, bw, bh = map(float, values[1:5])
                except ValueError as e:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: invalid label values: {values}"
                    ) from e

                if class_value != class_id:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: class_id must be an integer"
                    )
                if not 0 <= class_id < self.nc:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: class_id {class_id} "
                        f"outside [0, {self.nc - 1}]"
                    )
                if not all(np.isfinite(v) for v in (xc, yc, bw, bh)):
                    raise ValueError(
                        f"{label_path.name}:{line_number}: non-finite box values"
                    )
                if bw <= 0 or bh <= 0:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: "
                        "box width/height must be positive"
                    )

                x1 = np.clip((xc - bw / 2.0) * width, 0, width)
                y1 = np.clip((yc - bh / 2.0) * height, 0, height)
                x2 = np.clip((xc + bw / 2.0) * width, 0, width)
                y2 = np.clip((yc + bh / 2.0) * height, 0, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _load_mask(self, mask_path, image_size):
        mask_array = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()
        if mask_array.ndim != 2:
            raise RuntimeError(
                f"Expected single-channel mask, got {mask_array.shape}: {mask_path}"
            )
        if self.validate_mask:
            unexpected = set(np.unique(mask_array).tolist()) - set(range(self.nc + 1))
            if unexpected:
                raise ValueError(
                    f"{mask_path.name} contains unexpected mask values "
                    f"{sorted(unexpected)}; expected 0..{self.nc}. "
                    "If 255 is an ignore label, update the dataset convention "
                    "and loss accordingly."
                )
        expected_w, expected_h = image_size
        if mask_array.shape != (expected_h, expected_w):
            raise ValueError(
                f"Raw/mask size mismatch for {mask_path.name}: "
                f"raw={(expected_h, expected_w)}, mask={mask_array.shape}"
            )
        return torch.from_numpy(mask_array).long()

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["raw"]).convert("RGB")
        width, height = image.size
        target = {
            "boxes": _empty_boxes(),
            "labels": _empty_labels(),
            "instance_boxes": _empty_boxes(),
            "instance_labels": _empty_labels(),
            "instance_masks": _empty_instance_masks(height, width),
            "mask": torch.full((height, width), 255, dtype=torch.long),
            "has_semantic_mask": False,
            "has_instance_annotation": False,
            "has_instance_masks": False,
        }

        if self.task in ("detection", "both"):
            target.update(self._load_yolo_target(sample["label"], width, height))

        if self.task in ("segmentation", "both"):
            target["mask"] = self._load_mask(sample["mask"], image.size)
            target["has_semantic_mask"] = True

        target["has_instance_annotation"] = False
        target["has_instance_masks"] = False

        _validate_boxes_and_labels(target, "HqImprove1022")
        image, target = _resize_image_and_target(image, target, self.image_size)
        target["image_id"] = sample["id"]
        return _image_to_tensor(image), target


class UnifiedDetectionSegmentationDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError("datasets must be a non-empty list or tuple")

        self.datasets = list(datasets)
        self.dataset_offsets = []
        self.dataset_class_ranges = []
        self.global_class_names = []
        self.class_mapping = {}

        offset = 0
        for i, dataset in enumerate(self.datasets):
            if not isinstance(dataset, Dataset):
                raise TypeError(f"datasets[{i}] is not a Dataset instance")
            if getattr(dataset, "task", None) != "both":
                raise ValueError(
                    f"{type(dataset).__name__} must be instantiated with task='both'"
                )
            if isinstance(dataset, PascalVoc2012):
                source_name = "PascalVoc2012"
                names = list(VOC_CLASSES)
            elif isinstance(dataset, HqImprove1022):
                source_name = "HqImprove1022"
                names = list(dataset.class_names)
            else:
                raise TypeError(
                    "Supported source datasets: PascalVoc2012 and HqImprove1022; "
                    f"got {type(dataset).__name__}"
                )

            self.dataset_offsets.append(offset)
            class_start = offset
            class_end = offset + len(names) - 1
            self.dataset_class_ranges.append((class_start, class_end))

            for local_id, name in enumerate(names):
                global_id = offset + local_id
                qualified_name = f"{source_name}:{name}"
                self.global_class_names.append(qualified_name)
                self.class_mapping[qualified_name] = {
                    "dataset": source_name,
                    "local_id": local_id,
                    "global_id": global_id,
                    "mask_id": global_id + 1,
                }

            offset += len(names)

        self.num_detection_classes = offset
        self.num_segmentation_classes = offset + 1

        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

        print("[UnifiedDetectionSegmentationDataset]")
        print(f"  total samples: {len(self)}")
        print(f"  detection classes: {self.num_detection_classes}")
        print(
            f"  segmentation classes: {self.num_segmentation_classes} "
            "(including background)"
        )
        for ds, ds_offset, class_range in zip(
            self.datasets, self.dataset_offsets, self.dataset_class_ranges
        ):
            print(
                f"  {type(ds).__name__}: offset={ds_offset}, "
                f"global detection IDs={class_range[0]}..{class_range[1]}"
            )

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _locate_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        previous_end = 0
        for dataset_index, end in enumerate(self.cumulative_sizes):
            if index < end:
                return dataset_index, index - previous_end
            previous_end = end
        raise IndexError(index)

    def __getitem__(self, index):
        dataset_index, local_index = self._locate_index(index)
        source_dataset = self.datasets[dataset_index]
        image, target = source_dataset[local_index]

        required = {
            "boxes", "labels", "instance_boxes",
            "instance_labels", "instance_masks", "mask",
        }
        missing = required - set(target)
        if missing:
            raise KeyError(
                f"{type(source_dataset).__name__} did not return {sorted(missing)}"
            )

        offset = self.dataset_offsets[dataset_index]
        class_start, class_end = self.dataset_class_ranges[dataset_index]
        source_class_count = class_end - class_start + 1

        boxes = target["boxes"]
        labels = target["labels"].clone().long()
        instance_boxes = target["instance_boxes"]
        instance_labels = target["instance_labels"].clone().long()
        instance_masks = target["instance_masks"]
        mask = target["mask"].clone().long()

        if labels.numel():
            if int(labels.min()) < 0 or int(labels.max()) >= source_class_count:
                raise ValueError(
                    f"Invalid local detection labels "
                    f"{torch.unique(labels).tolist()} "
                    f"for {type(source_dataset).__name__}"
                )
            labels += offset

        if instance_labels.numel():
            if (
                int(instance_labels.min()) < 0
                or int(instance_labels.max()) >= source_class_count
            ):
                raise ValueError(
                    f"Invalid local instance labels "
                    f"{torch.unique(instance_labels).tolist()} "
                    f"for {type(source_dataset).__name__}"
                )
            instance_labels += offset

        instance_count = instance_labels.shape[0]
        if instance_boxes.shape != (instance_count, 4):
            raise ValueError("instance_boxes / instance_labels count mismatch")
        if instance_masks.ndim != 3:
            raise ValueError(
                "instance_masks must have shape [M,H,W], "
                f"got {tuple(instance_masks.shape)}"
            )
        if instance_masks.shape[0] != instance_count:
            raise ValueError("instance_masks / instance_labels count mismatch")
        if instance_masks.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                "instance mask/image size mismatch: "
                f"{tuple(instance_masks.shape[-2:])} vs "
                f"{tuple(image.shape[-2:])}"
            )

        if mask.ndim != 2:
            raise ValueError(f"semantic mask must be [H,W], got {tuple(mask.shape)}")
        if mask.shape != image.shape[-2:]:
            raise ValueError(
                "semantic mask/image size mismatch: "
                f"{tuple(mask.shape)} vs {tuple(image.shape[-2:])}"
            )

        foreground = (mask != 0) & (mask != 255)
        if torch.any(foreground):
            values = torch.unique(mask[foreground])
            if int(values.min()) < 1 or int(values.max()) > source_class_count:
                raise ValueError(
                    f"Invalid source mask values {values.tolist()} for "
                    f"{type(source_dataset).__name__}; "
                    f"expected 1..{source_class_count}"
                )
            mask[foreground] += offset

        target["boxes"] = boxes
        target["labels"] = labels
        target["instance_boxes"] = instance_boxes
        target["instance_labels"] = instance_labels
        target["instance_masks"] = instance_masks
        target["mask"] = mask
        target["has_instance_masks"] = bool(instance_count > 0)
        target["source_dataset"] = type(source_dataset).__name__
        target["source_index"] = local_index
        target["global_dataset_index"] = index
        return image, target

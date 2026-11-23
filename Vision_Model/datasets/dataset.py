"""
dataset_loaders.py

两个数据集加载类：
    1. PascalVoc2012
    2. HqImprove1022

统一返回格式：
    image: FloatTensor [3, H, W], RGB, range [0, 1]
    target: dict
        boxes: FloatTensor [N, 4], xyxy pixel coordinates
        labels: LongTensor [N], foreground class IDs (no background class)
        mask: LongTensor [H, W], optional semantic mask

类别约定：
    Detection:
        background is implicit; foreground IDs start at 0.
    Segmentation:
        background = 0
        foreground IDs start at 1
        VOC void/ignore pixels = 255

HqImprove1022:
    Detection class IDs: 0..5 (from YOLO txt)
    Segmentation mask IDs: 0=background, 1..6=foreground
    Therefore mask class ID = detection class ID + 1.

VOC2012:
    Detection IDs: 0..19
    Segmentation IDs: 0=background, 1..20=foreground, 255=ignore
"""

from pathlib import Path
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
    """
    适用于不同图像尺寸、不同目标数量的检测/分割任务。

    返回：
        images: list[Tensor]
        targets: list[dict]

    这样不会因为每张图的目标数量不同而在默认 collate 时出错。
    若设置了统一 image_size，后续也可以自行编写 stack 版本。
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def _image_to_tensor(image):
    """PIL RGB image -> float32 Tensor [3,H,W], [0,1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    return tensor


def _resize_image_and_target(image, target, image_size):
    """
    同步缩放图像、BBox 和 Mask。
    image_size 为 (height, width)。
    BBox 保持 xyxy 像素坐标，按 x/y 比例分别缩放。
    Mask 使用最近邻插值，避免产生新的类别值。
    """
    if image_size is None:
        return image, target

    new_h, new_w = image_size
    old_w, old_h = image.size

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if "boxes" in target and target["boxes"].numel() > 0:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] *= new_w / old_w
        boxes[:, [1, 3]] *= new_h / old_h
        target["boxes"] = boxes

    if "mask" in target:
        mask_array = target["mask"].cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask_array)
        mask_image = mask_image.resize(
            (new_w, new_h), Image.Resampling.NEAREST
        )
        target["mask"] = torch.from_numpy(
            np.asarray(mask_image, dtype=np.uint8).copy()
        ).long()

    return image, target


class PascalVoc2012(Dataset):
    """
    PASCAL VOC 2012 Dataset.

    Expected root:
        VOCdevkit/
            VOC2012/
                JPEGImages/
                Annotations/
                SegmentationClass/
                ImageSets/
                    Main/train.txt
                    Main/val.txt
                    Segmentation/train.txt
                    Segmentation/val.txt

    Args:
        root:
            Either the VOC2012 directory itself, or its parent directory
            containing VOCdevkit/VOC2012.
        split:
            "train" or "val".
        task:
            "detection", "segmentation", or "both".
            "both" uses the intersection of detection and segmentation IDs.
        image_size:
            None, or (height, width). If set, image/boxes/mask are resized
            consistently.
        include_difficult:
            Whether to include XML objects marked difficult=1.
        return_masks:
            Deprecated-style convenience override is intentionally omitted;
            use task="segmentation" or task="both".
    """

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
            raise ValueError(
                "task must be 'detection', 'segmentation', or 'both'"
            )

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
        self.num_detection_classes = len(VOC_CLASSES)       # 20 foreground classes
        self.num_segmentation_classes = len(VOC_CLASSES) + 1  # background + 20 classes

        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        self.mask_dir = self.root / "SegmentationClass"

        self.detection_ids = self._read_split_ids(
            self.root / "ImageSets" / "Main" / f"{split}.txt"
        )
        self.segmentation_ids = self._read_split_ids(
            self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        )

        if task == "detection":
            self.ids = self.detection_ids
        elif task == "segmentation":
            self.ids = self.segmentation_ids
        else:
            self.ids = sorted(
                set(self.detection_ids) & set(self.segmentation_ids)
            )

        if not self.ids:
            raise RuntimeError(
                f"No samples found for split={split}, task={task}. "
                "Check VOC2012 installation and ImageSets files."
            )

        print(
            f"[PascalVoc2012] split={split}, task={task}, "
            f"samples={len(self.ids)}"
        )

    @staticmethod
    def _read_split_ids(txt_path):
        if not txt_path.exists():
            raise FileNotFoundError(f"Split file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            return [line.strip().split()[0] for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _load_detection_target(self, image_id):
        xml_path = self.annotation_dir / f"{image_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"VOC annotation missing: {xml_path}")

        root = ET.parse(xml_path).getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            difficult = int(obj.findtext("difficult", default="0"))

            if difficult == 1 and not self.include_difficult:
                continue

            if name not in VOC_CLASS_TO_ID:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            # VOC XML coordinates are 1-based.
            xmin = float(bndbox.findtext("xmin")) - 1.0
            ymin = float(bndbox.findtext("ymin")) - 1.0
            xmax = float(bndbox.findtext("xmax")) - 1.0
            ymax = float(bndbox.findtext("ymax")) - 1.0

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASS_TO_ID[name])

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def _load_segmentation_mask(self, image_id):
        mask_path = self.mask_dir / f"{image_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"VOC segmentation mask missing: {mask_path}")

        # VOC SegmentationClass PNG is palette-indexed.
        # np.asarray(P-mode image) preserves class indices; do NOT convert to RGB.
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()

        if mask.ndim != 2:
            raise RuntimeError(
                f"Expected 2D VOC class mask, got shape={mask.shape}: {mask_path}"
            )

        # VOC: 0=background, 1..20=classes, 255=void/ignore.
        return torch.from_numpy(mask).long()

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"VOC image missing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        target = {}

        if self.task in ("detection", "both"):
            target.update(self._load_detection_target(image_id))

        if self.task in ("segmentation", "both"):
            target["mask"] = self._load_segmentation_mask(image_id)

        image, target = _resize_image_and_target(
            image, target, self.image_size
        )

        return _image_to_tensor(image), target


class HqImprove1022(Dataset):
    """
    User's custom Improve_Dataset_1022 dataset.

    Expected root:
        Improve_Dataset_1022/
            data.yaml
            train/
                raw/images_1.png
                label/labels_1.txt
                mask/masks_1.png
                cropped/             # ignored
            val/
                raw/
                label/
                mask/

    YOLO label format per line:
        class_id x_center y_center width height

    Class convention:
        detection labels: 0..nc-1
        mask: 0=background, class_id+1 for foreground.
    """

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
            raise ValueError(
                "task must be 'detection', 'segmentation', or 'both'"
            )

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
            self.class_names = [str(names.get(k, names.get(str(k)))) for k in keys]
        else:
            raise ValueError("data.yaml 'names' must be a list or dict")

        self.nc = int(config.get("nc", len(self.class_names)))
        self.num_detection_classes = self.nc
        self.num_segmentation_classes = self.nc + 1  # background + foreground classes

        if len(self.class_names) != self.nc:
            raise ValueError(
                f"data.yaml nc={self.nc}, but names has "
                f"{len(self.class_names)} entries"
            )

        self.split_dir = self.root / split
        self.raw_dir = self.split_dir / "raw"
        self.label_dir = self.split_dir / "label"
        self.mask_dir = self.split_dir / "mask"

        for directory in (self.raw_dir, self.label_dir, self.mask_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Dataset directory missing: {directory}")

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.samples = []

        raw_files = [
            p for p in self.raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        def natural_key(path):
            import re
            return [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", path.name)
            ]

        raw_files.sort(key=natural_key)

        for raw_path in raw_files:
            import re
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
        boxes = []
        labels = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                if len(values) < 5:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: expected "
                        "class xc yc w h"
                    )

                class_id = int(float(values[0]))
                xc, yc, bw, bh = map(float, values[1:5])

                if not 0 <= class_id < self.nc:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: class_id "
                        f"{class_id} outside [0, {self.nc - 1}]"
                    )

                if bw < 0 or bh < 0:
                    raise ValueError(
                        f"{label_path.name}:{line_number}: negative box size"
                    )

                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height

                x1 = np.clip(x1, 0, width)
                y1 = np.clip(y1, 0, height)
                x2 = np.clip(x2, 0, width)
                y2 = np.clip(y2, 0, height)

                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def _load_mask(self, mask_path, image_size):
        # Preserve grayscale class-index values. Do not convert to RGB.
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()

        if mask.ndim != 2:
            raise RuntimeError(
                f"Expected single-channel mask, got {mask.shape}: {mask_path}"
            )

        # User's convention: 0=background, 1..nc=foreground classes.
        valid_values = set(np.unique(mask).tolist())
        allowed_values = set(range(self.nc + 1))

        if self.validate_mask:
            unexpected = valid_values - allowed_values
            if unexpected:
                raise ValueError(
                    f"{mask_path.name} contains unexpected mask values "
                    f"{sorted(unexpected)}; expected 0..{self.nc}. "
                    "If 255 is an ignore label, explicitly add it to the "
                    "dataset convention and loss ignore_index."
                )

        mask_tensor = torch.from_numpy(mask).long()

        expected_w, expected_h = image_size
        if mask_tensor.shape != (expected_h, expected_w):
            raise ValueError(
                f"Raw/mask size mismatch for {mask_path.name}: "
                f"raw={(expected_h, expected_w)}, "
                f"mask={tuple(mask_tensor.shape)}"
            )

        return mask_tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["raw"]).convert("RGB")
        width, height = image.size

        target = {}

        if self.task in ("detection", "both"):
            target.update(
                self._load_yolo_target(sample["label"], width, height)
            )

        if self.task in ("segmentation", "both"):
            target["mask"] = self._load_mask(
                sample["mask"], image.size
            )

        image, target = _resize_image_and_target(
            image, target, self.image_size
        )

        return _image_to_tensor(image), target



class UnifiedDetectionSegmentationDataset(Dataset):
    """Merge instantiated PascalVoc2012 and HqImprove1022 datasets.

    Global detection IDs are contiguous from 0. Segmentation masks use
    0=background, 1..K=global foreground class ID + 1, and 255=ignore.
    Source datasets must be created with task="both".
    """

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
            self.dataset_class_ranges.append((offset, offset + len(names) - 1))

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
        self.num_segmentation_classes = offset + 1  # includes background

        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

        print("[UnifiedDetectionSegmentationDataset]")
        print(f"  total samples: {len(self)}")
        print(f"  detection classes: {self.num_detection_classes}")
        print(f"  segmentation classes: {self.num_segmentation_classes} (including background)")
        for ds, offset, class_range in zip(
            self.datasets, self.dataset_offsets, self.dataset_class_ranges
        ):
            print(
                f"  {type(ds).__name__}: offset={offset}, "
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

        required = {"boxes", "labels", "mask"}
        missing = required - set(target)
        if missing:
            raise KeyError(
                f"{type(source_dataset).__name__} did not return {sorted(missing)}; "
                "instantiate it with task='both'"
            )

        offset = self.dataset_offsets[dataset_index]
        class_start, class_end = self.dataset_class_ranges[dataset_index]
        source_class_count = class_end - class_start + 1

        boxes = target["boxes"]
        labels = target["labels"].clone().long()
        mask = target["mask"].clone().long()

        # Detection labels: source-local 0..C-1 -> global 0..K-1.
        if labels.numel():
            if int(labels.min()) < 0 or int(labels.max()) >= source_class_count:
                raise ValueError(
                    f"Invalid local detection labels {torch.unique(labels).tolist()} "
                    f"for {type(source_dataset).__name__}"
                )
            labels += offset

        # Semantic mask: 0 remains background; 1..C shift by the same offset.
        # VOC's 255 void pixels remain 255.
        foreground = (mask != 0) & (mask != 255)
        if torch.any(foreground):
            values = torch.unique(mask[foreground])
            if int(values.min()) < 1 or int(values.max()) > source_class_count:
                raise ValueError(
                    f"Invalid source mask values {values.tolist()} for "
                    f"{type(source_dataset).__name__}; expected 1..{source_class_count}"
                )
            mask[foreground] += offset

        target["boxes"] = boxes
        target["labels"] = labels
        target["mask"] = mask
        target["source_dataset"] = type(source_dataset).__name__
        target["source_index"] = local_index
        return image, target



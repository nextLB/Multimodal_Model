
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from torchvision.datasets import VOCDetection, VOCSegmentation


# ============================================================
# 1. VOC 2012 类别
# ============================================================

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# ============================================================
# 2. 生成 VOC 颜色映射表
# ============================================================

def create_voc_colormap(num_classes=256):

    colormap = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):

        r = 0
        g = 0
        b = 0

        index = i

        for j in range(8):

            r |= ((index >> 0) & 1) << (7 - j)
            g |= ((index >> 1) & 1) << (7 - j)
            b |= ((index >> 2) & 1) << (7 - j)

            index >>= 3

        colormap[i] = [r, g, b]

    return colormap


VOC_COLORMAP = create_voc_colormap()


# ============================================================
# 3. 处理 Torchvision VOC XML 标注结构
# ============================================================

def unwrap(value):

    while isinstance(value, list) and len(value) == 1:
        value = value[0]

    return value


def normalize_objects(objects):

    objects = unwrap(objects)

    if objects is None:
        return []

    if isinstance(objects, dict):
        return [objects]

    return objects


# ============================================================
# 4. 目标检测可视化
# ============================================================

def visualize_detection(image, target, save_path):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    annotation = target["annotation"]

    objects = normalize_objects(
        annotation.get("object", [])
    )

    print("\n========== Detection Annotation ==========")

    print("Image size:", image.size)
    print("Object count:", len(objects))

    for obj in objects:

        name = unwrap(obj.get("name", "unknown"))

        bbox = unwrap(obj.get("bndbox", {}))

        xmin = int(float(unwrap(bbox["xmin"])))
        ymin = int(float(unwrap(bbox["ymin"])))
        xmax = int(float(unwrap(bbox["xmax"])))
        ymax = int(float(unwrap(bbox["ymax"])))

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline="red",
            width=3
        )

        draw.text(
            (xmin, max(0, ymin - 15)),
            str(name),
            fill="yellow"
        )

        print(
            f"Class: {name:15s} "
            f"BBox: [{xmin}, {ymin}, {xmax}, {ymax}]"
        )

    image.save(save_path)

    print("\nDetection visualization saved:")
    print(save_path)


# ============================================================
# 5. 语义分割可视化
# ============================================================

def visualize_segmentation(image, mask, output_dir):

    image_np = np.array(image)

    mask_np = np.array(mask)

    # 检查 Mask 中实际存在的类别
    unique_ids = np.unique(mask_np)

    print("\n========== Segmentation Annotation ==========")

    print("Image size:", image.size)
    print("Mask size:", mask.size)

    print("Mask category IDs:")

    for category_id in unique_ids:

        if category_id < len(VOC_CLASSES):

            category_name = VOC_CLASSES[category_id]

        elif category_id == 255:

            category_name = "ignore"

        else:

            category_name = "unknown"

        print(
            f"ID: {category_id:3d} "
            f"Class: {category_name}"
        )

    # 将类别 ID 转换成 RGB 颜色
    mask_rgb = VOC_COLORMAP[mask_np]

    # VOC 的 255 表示 Ignore Label
    mask_rgb[mask_np == 255] = [128, 128, 128]

    # 生成叠加图
    overlay = image_np.astype(np.float32).copy()

    # 只对前景类别进行叠加，背景不覆盖原图
    foreground = (
        (mask_np > 0) &
        (mask_np < len(VOC_CLASSES))
    )

    overlay[foreground] = (
        0.55 * image_np[foreground] +
        0.45 * mask_rgb[foreground]
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    # 保存原始类别 ID Mask
    mask.save(
        output_dir / "segmentation_mask_index.png"
    )

    # 保存彩色 Mask
    Image.fromarray(mask_rgb).save(
        output_dir / "segmentation_mask_color.png"
    )

    # 保存叠加图
    Image.fromarray(overlay).save(
        output_dir / "segmentation_overlay.png"
    )

    # 绘制对比图
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    save_path = output_dir / "segmentation_comparison.png"

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    print("\nSegmentation visualization saved:")
    print(save_path)


# ============================================================
# 6. 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="VOC 数据集保存目录"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/voc_preview",
        help="可视化结果保存目录"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="选择第几张图片进行可视化"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)

    output_dir = Path(args.output_dir)

    data_root.mkdir(
        parents=True,
        exist_ok=True
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PASCAL VOC 2012 Dataset Downloader")
    print("=" * 60)

    print("Dataset root:", data_root.resolve())
    print("Output root:", output_dir.resolve())

    # --------------------------------------------------------
    # 下载并读取目标检测数据
    # --------------------------------------------------------

    print("\n[1/4] Downloading Detection Dataset...")

    detection_dataset = VOCDetection(
        root=str(data_root),
        year="2012",
        image_set="trainval",
        download=True
    )

    print("Detection samples:", len(detection_dataset))

    # --------------------------------------------------------
    # 下载并读取语义分割数据
    # --------------------------------------------------------

    print("\n[2/4] Loading Segmentation Dataset...")

    segmentation_dataset = VOCSegmentation(
        root=str(data_root),
        year="2012",
        image_set="trainval",
        download=True
    )

    print("Segmentation samples:", len(segmentation_dataset))

    # --------------------------------------------------------
    # 检查数据集
    # --------------------------------------------------------

    print("\n[3/4] Checking Dataset...")

    assert len(detection_dataset) > 0

    assert len(segmentation_dataset) > 0

    assert 0 <= args.index < len(detection_dataset)

    assert 0 <= args.index < len(segmentation_dataset)

    image_det, target_det = detection_dataset[args.index]

    image_seg, mask_seg = segmentation_dataset[args.index]

    assert image_det.size[0] > 0

    assert image_seg.size == mask_seg.size

    print("Detection image:", image_det.size)

    print("Segmentation image:", image_seg.size)

    print("Mask image:", mask_seg.size)

    print("Dataset verification passed!")

    # --------------------------------------------------------
    # 可视化目标检测
    # --------------------------------------------------------

    print("\n[4/4] Visualizing Dataset...")

    visualize_detection(
        image_det,
        target_det,
        output_dir / "detection_visualization.jpg"
    )

    # --------------------------------------------------------
    # 可视化语义分割
    # --------------------------------------------------------

    visualize_segmentation(
        image_seg,
        mask_seg,
        output_dir
    )

    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED!")
    print("=" * 60)

    print("Dataset:", data_root.resolve())

    print("Visualizations:", output_dir.resolve())


if __name__ == "__main__":
    main()

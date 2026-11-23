
from ultralytics import YOLO

# 1. 加载已经下载的预训练模型
yolo = YOLO("yolo26n-seg.pt")

# 2. 获取真正的 PyTorch 网络
net = yolo.model

print("=" * 80)
print("模型整体信息")
print("=" * 80)

print("模型类型:", type(net).__name__)
print("网络层容器:", type(net.model).__name__)
print("网络层数量:", len(net.model))

# 3. 遍历网络中的每一层
print("\n" + "=" * 80)
print("YOLO26n-seg 网络层结构")
print("=" * 80)

for i, layer in enumerate(net.model):
    # 模块类型
    module_type = type(layer).__name__

    # 参数量
    params = sum(p.numel() for p in layer.parameters())

    # 输入来源，-1 通常代表上一层
    from_layer = getattr(layer, "f", None)

    print(
        f"Layer {i:03d} | "
        f"From: {str(from_layer):12s} | "
        f"Type: {module_type:20s} | "
        f"Params: {params:,}"
    )



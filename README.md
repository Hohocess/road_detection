# LaneDetectionSystem

基于 YOLO 实例分割的车道识别项目，支持图片与视频处理，并提供 Windows 桌面窗口。

## 功能

- 图片车道识别与结果保存
- 视频逐帧车道识别、实时预览和结果视频导出
- 基于前序帧的车道轨迹平滑与短时预测，降低视频画面抖动
- 对分割掩膜拟合直线或二次曲线，支持弯道标记
- 模型短暂漏检时自动回退到最近的有效轨迹；没有历史轨迹时使用传统视觉检测回退

## 环境要求

- Windows 10/11
- Python 3.10（推荐）
- NVIDIA GPU 为可选项；没有 GPU 时可使用 CPU 推理，但视频处理速度会较慢

## 安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirement.txt
```

> PyTorch 的 CPU/GPU 安装包与 CUDA 版本有关。如果需要 GPU 推理，请根据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 `torch` 与 `torchvision`，再安装其余依赖。

## 模型文件

默认使用分割模型 `models/best.pt`。由于模型权重、数据集和训练输出体积较大，它们被 `.gitignore` 排除，不会随代码仓库提交。

请在运行前将训练好的权重放到：

```text
models/best.pt
```

模型需要包含名为 `lane` 的车道分割类别。项目的训练数据配置见 `lane.yaml`。

## Windows 桌面程序

启动窗口：

```powershell
python window.py
```

使用方法：

1. 点击“选择图片”或“选择视频”。
2. 图片会立即识别，并保存为 `原文件名_lane.jpg`。
3. 视频点击“开始识别”后会逐帧预览；勾选“保存识别视频”时，输出为 `原文件名_lane.mp4`。
4. 处理新图片或视频时，历史车道轨迹会自动清空，避免不同来源之间相互影响。

支持的视频格式：`mp4`、`avi`、`mov`、`mkv`。

## 命令行图片识别

```powershell
python main.py -i 00240.jpg -o result.jpg -s
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `-i`, `--input` | 输入图片路径 |
| `-o`, `--output` | 输出图片路径，默认为 `result.jpg` |
| `-s`, `--show` | 保存后显示处理结果 |
| `--classical` | 不使用 YOLO 分割，仅使用传统检测回退方案 |

## 项目结构

```text
LaneDetectionSystem/
├── window.py          # PyQt5 图片/视频桌面程序
├── main.py            # 命令行图片识别入口
├── cv2_process.py     # 轨迹拟合、曲线绘制与时序平滑
├── yolo_process.py    # YOLO 分割模型加载与推理
├── train.py           # 训练续训入口
├── lane.yaml          # 数据集配置
├── requirement.txt    # Python 依赖
└── models/best.pt     # 训练权重（需自行提供，不提交到 Git）
```

## 训练

准备好符合 `lane.yaml` 配置的数据集后，可通过以下命令继续已有训练：

```powershell
python train.py
```

训练脚本当前从 `runs/segment/CULane_YOLOv8_seg-2/weights/last.pt` 继续训练。请根据自己的权重路径和数据集位置调整 `train.py` 与 `lane.yaml`。

## 常见问题

### 找不到 `models/best.pt`

将训练好的模型放入 `models` 目录并命名为 `best.pt`，或在代码中将 `LaneDetection` 的 `model_path` 修改为实际路径。

### 视频速度较慢

优先使用 CUDA 版本的 PyTorch 和 NVIDIA GPU；也可适当降低输入视频分辨率。CPU 推理可以正常使用，但帧率会明显降低。

### 车道线偶尔跳动或消失

系统会用历史帧对短时漏检进行预测。若场景、摄像头位置或道路类型和训练集差别较大，建议补充对应场景数据并重新训练分割模型。

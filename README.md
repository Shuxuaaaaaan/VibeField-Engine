# VibeField-Engine

[English](#english) | [中文](#chinese)

---
<a name="english"></a>
## 🇬🇧 English

VibeField-Engine is a high-performance CLI tool for extracting physical vibration vector fields from Eulerian Video Magnification (EVM) or Phase-Based Video Magnification (PVM) amplified videos. It avoids blind per-pixel Fast Fourier Transforms (FFT) by implementing a robust pipeline consisting of saliency detection, patch-based spatial aggregation, and complex-domain FFT, resulting in increased processing speed and enhanced noise suppression. The output is compactly stored using the Zarr format, allowing scalable and chunked access to multi-dimensional data.

### Requirements

- Python 3.10+
- `uv` package manager

### Installation

Install dependencies using `uv`:

```bash
uv sync
```

### Usage

Run the CLI tool via `uv` or directly if the environment is activated:

```bash
uv run python src/cli.py -i data/resources/input_video.mp4 -p 8 -f 10,50 -s 0.01
```

#### CLI Parameters

- `-i, --input`: (Required) Path to the input video file or directory containing videos.
- `-p, --patch-size`: (Optional) Size of the feature block $N \times N$ for spatial aggregation. Default to `1` (per-pixel).
- `-f, --freq-range`: (Optional) The frequency range of interest, comma-separated e.g. `10,50` (Hz).
- `-s, --saliency-threshold`: (Optional) Threshold for saliency filtering. Regions with subpixel variance below this threshold will be skipped. Default is a very small epsilon.
- `-o, --output-dir`: (Optional) Output directory for the `.zarr` files. Default is `data/results`.

### Visualization

To visualize the generated vector fields overlaid on the original video, use the interactive viewer:

```bash
uv run python src/viewer.py -i data/resources/input_video.mp4 -z data/results/input_video.zarr
```
- A GUI window will open playing the video.
- Red dots indicate the centers of the active patches.
- Blue arrows display the instantaneous vibration movement reconstructed from the extracted complex frequency domain components.
- Press `q` or `ESC` to close the viewer.

### GUI Analysis Tool

For a fully integrated graphical user interface that combining global computations, vector overlays, and localized Region of Interest (ROI) tracking (with waveform and orbit plots), use the `analysis.py` tool:

```bash
uv run src/analysis.py
```
- Select a video from the list (sourced from `data/resources/`).
- Use the right panel to execute **Global Computing** to generate the `.zarr` file.
- Toggle **Overlay Rendering** to preview vector fields directly dynamically.
- Click **Add ROI (+)** to draw localized tracking regions and view interactive pyqtgraph real-time charts (`X(t)`, `Y(t)`).

---
<a name="chinese"></a>
## 🇨🇳 中文

VibeField-Engine 是一个高性能的命令行工具，专门用于将欧拉视频放大 (EVM) 或基于相位的视频放大 (PVM) 处理后的视频，转换为物理震动矢量场。它通过引入显著性检测、基于图像块 (Patch) 的空间聚合以及复数域 FFT，避免了盲目的逐像素快速傅里叶变换，极大提升了处理速度兼具良好的噪声抑制能力。输出结果以 Zarr 格式高效存储。

### 环境要求

- Python 3.10+
- `uv` 包管理器

### 安装指导

使用 `uv` 安装项目及依赖：

```bash
uv sync
```

### 快速开始

使用 `uv run` 执行主程序：

```bash
uv run python src/cli.py -i data/resources/input_video.mp4 -p 8 -f 10,50 -s 0.01
```

#### CLI 参数详解

- `-i, --input`：(必填) 输入原始视频的文件路径，或包含多个视频的目录。
- `-p, --patch-size`：(可选) 特征块的空间聚合尺寸 $N \times N$（如 8, 16）。默认值为 `1`，即逐像素。
- `-f, --freq-range`：(可选) 感兴趣的频率范围（单位：Hz），格式如 `10,50`。
- `-s, --saliency-threshold`：(可选) 显著性过滤的方差阈值，低于该阈值的平静区域将被略过不处理。
- `-o, --output-dir`：(可选) 指定 Zarr 文件的保存路径。默认存储在 `data/results` 目录。

### 可视化工具 (Viewer)

生成矢量场文件后，可以使用自带的 GUI 工具在原视频上动态叠加震动箭头进行预览：

```bash
uv run python src/viewer.py -i data/resources/input_video.mp4 -z data/results/input_video.zarr
```
- 程序会弹出一个播放窗口与原视频同步播放。
- 红色的点代表具有活跃震动的特征块中心。
- 蓝色的动态箭头则利用提取到的时域频率与相位信息，实时重建并放大显示其物理运动轨迹。
### 综合可视化分析工具 (GUI Analysis Tool)

如果您希望拥有一个集成全局计算、动态矢量叠加渲染以及局部特征追踪（包含波形图与轨道图分析）的图形界面，请使用新加入的 `analysis.py` 工具：

```bash
uv run src/analysis.py
```
- 从列表中选择视频（视频来源于 `data/resources/`）。
- 使用右侧面板进行 **全局计算 (Global Computing)**，后台生成 `.zarr` 矢量文件。
- 勾选 **叠加渲染 (Overlay Rendering)** 即可在播放原视频时预览流体矢量动效。
- 点击 **Add ROI (+)** 在画面中绘制特定的兴趣追踪区域，可在随后生成的列表中打开交互式的 pyqtgraph 实时追踪图表界面（如 `X(t)`、`Y(t)` 曲线及震动轨道）。

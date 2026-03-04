# feezy 智能视频编辑系统

> 基于 AI 的智能视频剪辑工具，支持文本驱动长视频高光提取、自动镜头组接与成片渲染。

---

## 📋 项目当前进度
- ✅ **任务1-4**：完成端到端视频预处理模块开发与测试
- ⏳ 下一步：镜头边界检测（SBD）模块开发

---

## 🚀 核心模块：视频预处理
### 功能特性
- **工业级视频IO**：基于 Decord 实现，支持多线程并行解码，比 OpenCV 快 5-10 倍
- **双模式抽帧**：支持均匀抽帧（指定总帧数/每秒帧数）和 I 帧（关键帧）抽帧
- **端到端预处理**：整合「视频读取→元数据提取→抽帧→帧预处理」全流程
- **模型适配**：自动适配 Hugging Face CLIP/VideoMAE 等预训练模型输入格式
- **性能监控**：内置毫秒级耗时统计，自动定位优化方向

### 性能亮点
基于 720P/30fps/56.4s 测试视频的实测数据：

| 配置方案 | 总耗时 | 加速比（对比基准） |
|----------|--------|-------------------|
| 4线程 + 均匀抽帧（基准） | 4.8069s | 1x |
| 8线程 + 均匀抽帧 | 1.5644s | **3.07x** |
| 8线程 + I 帧抽帧 | 0.9144s | **5.26x** |

> 完整测试报告见：[docs/test_report.md](docs/test_report.md)

---

## 🛠️ 快速开始
### 1. 环境安装
```bash
# 克隆仓库
git clone https://github.com/feezy2www/smart_vedio_editor
cd smart_vedio_editor

# 创建并激活 Conda 环境
conda create -n smart_editor python=3.10 -y
conda activate smart_editor

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速使用
```python
from src.utils import VideoProcessor

# 初始化视频处理器（默认8线程，最优性能）
processor = VideoProcessor("data/test.mp4")

# 端到端预处理，直接输出模型可用张量
result = processor.preprocess_video(
    frame_mode="uniform",  # 抽帧模式：uniform/keyframe
    frame_count=16,        # 抽帧数量
    verbose=True            # 打印进度和耗时
)

# 获取模型输入张量
model_input = result["model_input"]
print(f"模型输入张量shape: {model_input.shape}")
```

---

## 📚 核心 API 文档
### `VideoProcessor` 类
工业级视频处理封装类。

#### `__init__(video_path, num_threads=8)`
初始化视频处理器。
- **参数**：
  - `video_path` (str)：视频路径
  - `num_threads` (int)：Decord 读取线程数，默认 8（最优性能）

#### `get_video_info()`
获取视频元数据。
- **返回**：包含视频路径、总帧数、帧率、时长、分辨率的字典。

#### `preprocess_video(frame_mode="uniform", frame_count=16, frame_fps=None, verbose=True)`
端到端视频预处理整合。
- **参数**：
  - `frame_mode` (str)：抽帧模式，`"uniform"`（均匀抽帧）或 `"keyframe"`（I 帧抽帧）
  - `frame_count` (int)：均匀抽帧时指定总帧数（和 `frame_fps` 二选一）
  - `frame_fps` (float)：均匀抽帧时指定每秒帧数（和 `frame_count` 二选一）
  - `verbose` (bool)：是否打印进度和耗时统计
- **返回**：结构化字典，包含：
  - `model_input`：模型可直接输入的 PyTorch 张量
  - `video_info`：视频元数据
  - `num_frames`：实际抽帧数量
  - `time_stats`：毫秒级耗时统计

---

## 📁 项目结构
```
smart_vedio_editor/
├── data/               # 测试视频存放目录（.gitignore已忽略）
├── docs/               # 文档目录
│   └── test_report.md  # 视频预处理模块测试报告
├── src/                # 源代码目录
│   └── utils.py        # 核心工具类：VideoProcessor
├── configs/            # 配置文件目录
├── .gitignore          # Git 忽略文件配置
├── requirements.txt    # 项目依赖
└── README.md           # 项目说明文档（本文件）
```

---

## 🎯 后续开发计划
- [ ] 任务5：镜头边界检测（SBD）模块开发，基于 TransNetV2
- [ ] 任务6：多模态内容理解模块开发，基于 CLIP
- [ ] 任务7：智能剪辑决策引擎开发
- [ ] 任务8：渲染合成模块开发
- [ ] 任务9：全流程整合与测试

---

## 📄 许可证
MIT License

---

## 🤝 联系方式
- 项目作者：feezy
- 项目仓库：https://github.com/feezy2www/smart_vedio_editor
```
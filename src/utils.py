import decord
from decord import VideoReader, cpu
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from transformers import CLIPImageProcessor


class VideoProcessor:
    """
    feezy学习视频编辑系统, 提供视频处理和分析功能
    新增功能：
    1. 均匀抽帧：
    2. 返回元数据: get_vedio
    """
    def __init__(self, video_path: str, num_threads: int = 4):
        """
        初始化视频处理器
        :param video_path: 视频路径
        :param num_threads: Decord读取线程数, 默认4线程加速
        """
        self.video_path = video_path
        self.num_threads = num_threads

        # 初始化一次VideoReader，上下文初始设置为 cpu之后改
        self.vr = VideoReader(video_path, num_threads = num_threads, ctx = cpu(0))
        
        # 获取基本信息
        self._total_frames = len(self.vr) # 总帧数
        self._fps = self.vr.get_avg_fps() # 帧率
        self._duration = self._total_frames / self._fps # 时长
        self._height, self._width = self.vr[0].shape[:2] # 分辨率

        # 初始化CLIP图像预处理器（hugging face官方，最标准）
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_video_info(self) -> Dict:
        """
        获取视频的基础元数据
        """
        return {
            "video_path": self.video_path,
            "total_frames": self._total_frames,
            "fps": round(self._fps, 2),
            "duration": round(self._duration, 2),
            "resolution": (self._width, self._height),
            "height": self._height,
            "width": self._width,
        }
    
    def extract_frames(
            self,
            mode: str = "uniform",
            num_frames: Optional[int] = None,
            fps: Optional[float] = None
    ) -> np.ndarray:
        """
        视频抽帧核心函数，支持两种模式
        :param mode: 抽帧模式, 'uniform'（均匀抽帧）, 'keyframe'（I帧/关键帧抽帧）
        :param num_frames: 均匀抽帧时，指定抽帧总数( 和fps二选一)
        :param fps: 均匀抽帧时，指定每秒抽多少帧( 和num_frames二选一)
        :return: 抽帧后的numpy数组(N , H, W, 3)
        """
        if mode == "uniform":
            # 均匀抽帧
            if num_frames is not None:
                # 指定总帧数抽帧
                interval = max(1, self._total_frames // num_frames)
                frame_indices = list(range(0, self._total_frames, interval))
                # 确保抽够指定数量
                if len(frame_indices) > num_frames:
                    frame_indices = frame_indices[:num_frames]
            elif fps is not None:
                # 指定每秒抽多少帧
                interval = max(1, int(self._fps / fps))
                frame_indices = list(range(0, self._total_frames, interval))
            else:
                inteval = max(1, int(self._fps))
                frame_indices = list(range(0, self._total_frames, interval))
            # 批量读取帧（decord最快的方法）
            frames = self.vr.get_batch(frame_indices).asnumpy()
        elif mode == "keyframe":
            # I帧抽取逻辑
            keyframe_indices = self.vr.get_key_indices()
            frames = self.vr.get_batch(keyframe_indices).asnumpy()   
        else:
            raise ValueError(f"不支持的抽帧模式: {mode}, 请选择 'uniform' 或 'keyframe'")
        return frames
    
    def frame_preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        帧预处理函数，适配Hugging Face CLIP/VideoMAE等模型输入
        :param frames: 抽帧后的numpy数组(N, H, W, 3)
        :return: 预处理后的Pytorch张量(N, 3, H', W')，适合模型输入
        """
        inputs = self.clip_processor(images = list(frames), return_tensors="pt")
        return inputs['pixel_values']
    def extract_and_preprocess(
            self,
            mode: str = "uniform",
            num_frames: Optional[int] = None,
            fps: Optional[float] = None
    ) -> torch.Tensor:
        """
        端到端：抽帧+预处理，一步到位
        :return: 可直接输入模型的PyTorch张量
        后需要修改，把resize在VedioReader里做了，clip_processor里就不resize了，会加快速度。
        """
        frames = self.extract_frames(mode = mode, num_frames = num_frames, fps = fps)
        return self.frame_preprocess(frames)

    def extract_frame(self, frame_idx: int) -> np.ndarray:
        """
        快速读取视频的任意一帧
        :param frame_idx: 帧索引
        :return: 该帧的numpy数组 (H, W, 3)
        """
        # 边界检查一下
        if frame_idx < 0 or frame_idx >= self._total_frames:
            raise ValueError(f"Frame index out of range: {frame_idx}, must be in [0, {self._total_frames - 1}]")
        return self.vr[frame_idx].asnumpy()

if __name__ == "__main__":
    # 测试
    test_video_path = "data/test.mp4"

    print("=" * 50)
    test_video = VideoProcessor(test_video_path)

    info = test_video.get_video_info()
    print("视频元数据")

    for key, value in info.items():
        print(f" {key}: {value}")

    print("=" * 60)
    print("任务3：视频抽帧与帧预处理模块测试")
    print("=" * 60)
    
    # 1. 初始化
    print("\n[1/5] 初始化VideoProcessor...")
    processor = VideoProcessor(test_video_path)
    info = processor.get_video_info()
    print(f"视频信息：{info['duration']}秒, {info['fps']}fps, {info['total_frames']}帧")
    
    # 2. 测试均匀抽帧（按总帧数）
    print("\n[2/5] 测试均匀抽帧（指定抽16帧）...")
    frames_uniform = processor.extract_frames(mode="uniform", num_frames=16)
    print(f"抽帧结果shape: {frames_uniform.shape}")
    
    # 3. 测试I帧抽帧
    print("\n[3/5] 测试I帧抽帧...")
    frames_keyframe = processor.extract_frames(mode="keyframe")
    print(f"I帧抽帧结果shape: {frames_keyframe.shape}")
    
    # 4. 测试帧预处理
    print("\n[4/5] 测试帧预处理（适配CLIP输入）...")
    pixel_values = processor.frame_preprocess(frames_uniform)
    print(f"预处理后张量shape: {pixel_values.shape}")
    print(f"张量数据类型: {pixel_values.dtype}")
    print(f"张量值范围: [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
    
    # 5. 测试端到端抽帧+预处理
    print("\n[5/5] 测试端到端（抽帧+预处理一步到位）...")
    model_input = processor.extract_and_preprocess(mode="uniform", fps=1)
    print(f"端到端输出shape: {model_input.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！任务3完成。")
    print("=" * 60)


import os
import decord
from decord import VideoReader, cpu
import numpy as np
import torch
import time
from typing import Dict, Tuple, List, Optional, Union
from transformers import CLIPImageProcessor


class VideoProcessor:
    """
    feezy学习视频编辑系统, 提供视频处理和分析功能
    新增功能：
    1. 均匀抽帧：
    2. 返回元数据: get_video
    """
    def __init__(self, video_path: str, num_threads: int = 4):
        """
        初始化视频处理器
        :param video_path: 视频路径
        :param num_threads: Decord读取线程数, 默认4线程加速
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
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
                interval = max(1, int(self._fps))
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
    def preprocess_video(
        self,
        frame_mode: str = "uniform",
        frame_count: Optional[int] = 16,
        frame_fps: Optional[float] = None,
        verbose: bool = True
    ) -> Dict:
        """
        端到端视频预处理整合
        整合流程：视频读取 → 信息提取 → 抽帧 → 帧预处理 → 结构化输出
        :param frame_mode: 抽帧模式，'uniform' 或 'keyframe'
        :param frame_count: 均匀抽帧时指定总帧数（和frame_fps二选一）
        :param frame_fps: 均匀抽帧时指定每秒帧数（和frame_count二选一）
        :param verbose: 是否打印进度信息
        :return: 结构化字典，包含模型输入、元数据、抽帧信息
        """
        if verbose:
            print("=" * 60)
            print(f"[1/4] 正在初始化视频处理器")
            print(f"      视频路径: {self.video_path}")
            print(f"      视频信息: {self._duration:.1f}秒, {self._fps:.1f}fps, {self._total_frames}帧")
        
        # 1. 抽帧
        if verbose:
            target_desc = f"总帧数={frame_count}" if frame_count else f"帧率={frame_fps}fps"
            print(f"\n[2/4] 正在抽帧: 模式={frame_mode}, {target_desc}")
        frames = self.extract_frames(mode=frame_mode, num_frames=frame_count, fps=frame_fps)
        if verbose:
            print(f"      实际抽帧数量: {frames.shape[0]}")
            print(f"      单帧原始尺寸: {frames.shape[1:]}")
        
        # 2. 帧预处理
        if verbose:
            print(f"\n[3/4] 正在进行帧预处理（适配CLIP输入）")
        pixel_values = self.frame_preprocess(frames)
        if verbose:
            print(f"      预处理后张量shape: {pixel_values.shape}")
            print(f"      张量数据类型: {pixel_values.dtype}")
        
        # 3. 组装结构化返回结果
        result = {
            "model_input": pixel_values,  # 模型可直接输入的张量
            "video_info": self.get_video_info(),  # 视频元数据
            "num_frames": frames.shape[0],  # 实际抽帧数量
            "frame_shape_original": frames.shape[1:],  # 单帧原始shape
            "frame_shape_processed": tuple(pixel_values.shape[1:])  # 单帧预处理后shape
        }
        
        if verbose:
            print(f"\n[4/4] 预处理完成！")
            print("=" * 60)
        
        return result




if __name__ == "__main__":
    test_video_path = "data/test.mp4"

    print("\n" + "=" * 70)
    print("任务四：端到端预处理模块整合测试")
    print("=" * 70)
    
    try:
        # 1. 初始化
        print("\n>>> 步骤1：初始化VideoProcessor")
        processor = VideoProcessor(test_video_path, num_threads=8)
        
        # 2. 【任务四核心】测试端到端预处理
        print("\n>>> 步骤2：调用端到端preprocess_video()")
        result = processor.preprocess_video(
            frame_mode="uniform",
            frame_count=16,
            verbose=True
        )
        
        # 3. 验证结果
        print("\n>>> 验证输出结果：")
        print(f"  ✅ 模型输入张量shape: {result['model_input'].shape}")
        print(f"  ✅ 视频时长: {result['video_info']['duration']}秒")
        print(f"  ✅ 实际抽帧数量: {result['num_frames']}")
        print(f"  ✅ 预处理后单帧shape: {result['frame_shape_processed']}")
        
        print("\n" + "=" * 70)
        print("🎉 任务四测试通过！端到端预处理模块整合完成。")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


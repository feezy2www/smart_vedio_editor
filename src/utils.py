import decord
from decord import VideoReader
import numpy as np
from typing import Dict, Tuple, List

class VideoProcessor:
    """
    feezy学习视频编辑系统, 提供视频处理和分析功能
    """
    def __init__(self, video_path: str, num_threads: int = 4):
        """
        初始化视频处理器
        :param video_path: 视频路径
        :param num_threads: Decord读取线程数, 默认4线程加速
        """
        self.video_path = video_path
        self.num_threads = num_threads

        # 初始化一次VideoReader，所有方法共享
        self.vr = VideoReader(video_path, num_threads = num_threads)
        
        # 获取基本信息
        self._total_frames = len(self.vr) # 总帧数
        self._fps = self.vr.get_avg_fps() # 帧率
        self._duration = self._total_frames / self._fps # 时长
        self._height, self._width = self.vr[0].shape[:2] # 分辨率

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

    def extract_frame_batch(self, frame_indices: List[int]) ->np.ndarray:
        """
        批量读取多帧（decord的核心优势，比循环单帧读取快10倍+）
        :param frame_indices: 帧索引列表
        :return: 批量帧的numpy数组 (N, H, W, 3) 一共N帧
        """
        return self.vr.get_batch(frame_indices).asnumpy()

    def extract_keyframes(self, interval: int = 10) -> np.ndarray:

        keyframes_indices = list(range(0, self._total_frames, interval))
        return self.extract_frame_batch(keyframes_indices)

if __name__ == "__main__":
    # 测试
    test_video_path = "data/test.mp4"

    print("=" * 50)
    test_video = VideoProcessor(test_video_path)

    info = test_video.get_video_info()
    print("视频元数据")

    for key, value in info.items():
        print(f" {key}: {value}")
    frame1 = test_video.extract_frame(521)
    print(frame1.shape)

    # frame_1 = test_video.extract_frame(-1)
    # print(frame_1)

    frame_v = test_video.extract_frame_batch([1, 10, 20, 100])
    print(f"批量抽帧{frame_v.shape}")

    frame_key = test_video.extract_keyframes(20)
    print(f"均匀抽帧{frame_key.shape}")
    

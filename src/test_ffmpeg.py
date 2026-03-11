import ffmpeg

video_path = "./data/test.mp4"  # 改成你的视频路径

print("正在测试ffmpeg调用（和transnetv2用的参数一样）...")
try:
    # 模拟transnetv2的ffmpeg调用（从transnetv2_pytorch.py第327行复制的参数）
    out, err = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='48x27')
        .run(capture_stdout=True, capture_stderr=True)
    )
    print("✅ ffmpeg调用成功！")
except ffmpeg.Error as e:
    print("❌ ffmpeg调用失败！")
    print("\nffmpeg 标准错误输出 (stderr):")
    print(e.stderr.decode('utf-8'))
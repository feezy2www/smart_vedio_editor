import torch
from transnetv2_pytorch import TransNetV2
from decord import VideoReader, cpu
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import clip
import os
from typing import List, Dict

def detect_shot_boundaries(video_path: str):
    """
    使用官方 TransNetV2 进行镜头边界检测
    """
    model = TransNetV2()

    # 高效 FFmpeg 读取和滑动窗口分块推理，（比自己用decord进行读取安全多了）
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    # 原始的video_frames 归一化的原始视频帧,[N, 3, H, W]（N = 总帧数，3=RGB 通道，H = 帧高度，W = 帧宽度）torch.Tensor（PyTorch 张量）归一化到 [0.0, 1.0]（模型输入要求，原始像素值 0-255 被除以 255
    # single_frame_predictions单帧级镜头边界预测结果,实际上是tensor，[N, 2]（N = 总帧数，2 = 两个预测值：「非边界概率」「边界概率」），[0.0, 1.0]（概率值，两列之和≈1）
    # ransNetV2 采用「滑动窗口」机制处理视频帧（窗口大小约 16 帧），all_frame_predictions 是窗口内每帧的原始预测得分，而 single_frame_predictions 是模型对这些得分做归一化、二分类后的简化结果 ——对于镜头检测的业务场景，你只需要关注 single_frame_predictions 即可

    if torch.is_tensor(single_frame_predictions):
        single_frame_predictions = single_frame_predictions.cpu().numpy()

    scenes = model.predictions_to_scenes(single_frame_predictions)

    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()

    shots = []

    for idx, (start_frame, end_frame) in enumerate(scenes):
        duration = (end_frame - start_frame + 1) / fps
        shots.append({
            "shot_id": idx + 1,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_time": round(start_frame / fps, 2),
            "end_time": round(end_frame / fps, 2),
            "duration": round(duration, 2)
        })

    return shots, vr, fps

def extract_shot_feature(video_path: str, shots: list, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    对每个镜头提取关键帧和CLIP特征
    """

    print(f"正在加载CLIP模型到{device}...")

    model, preprocess = clip.load("ViT-B/32", device = device)

    vr = VideoReader(video_path, ctx = cpu(0))
    
    structured_results = []
    
    print(f"开始处理{len(shots)}个镜头的特征提取")
    for shot in shots:
        # 1. 选取关键帧：区镜头中间帧
        mid_frame_idx = (shot["start_frame"] + shot["end_frame"]) // 2

        # 2. y用Decord读取关键帧：返回时[H, W, 3]的numpy数组
        frame = vr[mid_frame_idx].asnumpy()

        # 3. 转成PIL Image，适配CLIP输入
        pil_image = Image.fromarray(frame)

        # 4. CLIP 预处理 + 特征提取
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # 提取特征向量，并归一化
            image_feature = model.encode_image(image_input)
            # norm方法计算归一化结果
            image_feature /= image_feature.norm(dim = -1, keepdim = True)
        
        shot_struct = {
            "shot_id": shot["shot_id"],
            "metadata":{
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "start_time": shot["start_time"],
                "end_time": shot["end_time"],
                "duration": shot["duration"],
                "key_frame_idx": int(mid_frame_idx)
            },
            "clip_feature": image_feature.cpu().numpy().tolist()[0]
        }

        structured_results.append(shot_struct)

        if shot["shot_id"] % 10 == 0:
            print(f"已处理{shot['shot_id']}/{len(shots)} 个镜头...")
    print("处理完所有的镜头了")
    return structured_results

def save_sturcture_to_json(structured_data: list, output_path: str = "./output/structured_video.json"):
    """
    将结构化结果保存为JSON文件
    """
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    with open(output_path, 'w', encoding = 'utf-8') as f:
        json.dump(structured_data, f, indent = 2, ensure_ascii = False)
    print(f"结构化结果已保存至: {output_path}")

def video_structuring(video_path: str, output_json_path: str = "./output/structured_video.json") -> List[Dict]:
    """
    封装：整合「镜头检测 -> 特征提取 -> 结构化输出」全流程
    """
    print("="*60)
    print("步骤1/3：正在进行镜头边界检测...")
    print("="*60)
    shots, vr, fps = detect_shot_boundaries(video_path)
    
    print("\n" + "="*60)
    print("步骤2/3：正在提取镜头级CLIP特征...")
    print("="*60)
    structured_data = extract_shot_feature(video_path, shots)
    
    print("\n" + "="*60)
    print("步骤3/3：正在保存结构化JSON文件...")
    print("="*60)
    save_sturcture_to_json(structured_data, output_json_path)
    
    print("\n🎉 视频结构化全流程完成！")
    return structured_data

def visualize_shots(video_path: str, structured_data: List[Dict], output_image_path: str = "./output/shot_visualization.jpg", shots_per_row: int = 4):
    """
    可视化：将每个镜头的关键帧、时间、时长拼接成一张大图
    【优化1】文字完全放在缩略图下方，彻底不遮挡画面内容
    【优化2】等比例缩放图片
    """
    print("\n" + "="*60)
    print("正在生成镜头可视化图谱...")
    print("="*60)
    
    vr = VideoReader(video_path, ctx=cpu(0))
    cells = []  # 每个元素是「缩略图+下方文字」的完整单元
    
    # ========== 尺寸参数配置 ==========
    thumbnail_img_size = (320, 180)  # 纯画面的尺寸（宽, 高）16:9
    text_area_height = 80  # 下方文字区域的高度
    cell_total_width = thumbnail_img_size[0]
    cell_total_height = thumbnail_img_size[1] + text_area_height  # 每个单元的总高度
    
    # 尝试加载字体，失败则用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
            except:
                font = ImageFont.load_default(size=20)
    
    # 1. 生成每个「画面+下方文字」的完整单元
    for shot in structured_data:
        # 读取关键帧
        frame = vr[shot["metadata"]["key_frame_idx"]].asnumpy()
        pil_img = Image.fromarray(frame)
        
        # ========== 保持宽高比缩放画面，不变形 ==========
        original_w, original_h = pil_img.size
        target_w, target_h = thumbnail_img_size
        
        # 计算缩放比例，保持宽高比
        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        
        # 按比例缩放画面
        resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 创建纯黑画布，把画面居中贴进去
        img_canvas = Image.new("RGB", thumbnail_img_size, (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        img_canvas.paste(resized_img, (paste_x, paste_y))
        
        # ========== 创建「画面+文字」的完整单元 ==========
        # 新建单元画布：上方是画面，下方是浅灰色文字区域
        cell_canvas = Image.new("RGB", (cell_total_width, cell_total_height), (240, 240, 240))
        # 把画面贴到单元的顶部
        cell_canvas.paste(img_canvas, (0, 0))
        
        # ========== 在下方文字区域绘制文字（居中对齐，完全不碰画面） ==========
        draw = ImageDraw.Draw(cell_canvas)
        text = f"Shot {shot['shot_id']}\n{shot['metadata']['start_time']:.1f}s - {shot['metadata']['end_time']:.1f}s\n({shot['metadata']['duration']:.1f}s)"
        
        # 计算文字尺寸，实现水平+垂直居中
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 文字坐标：在文字区域里完全居中
        text_x = (cell_total_width - text_width) // 2
        text_y = thumbnail_img_size[1] + (text_area_height - text_height) // 2
        
        # 直接绘制黑色文字（有浅灰色背景，不需要额外黑框，清晰不刺眼）
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
        
        cells.append(cell_canvas)
    
    # 2. 把所有单元拼接成大图
    num_shots = len(cells)
    num_rows = (num_shots + shots_per_row - 1) // shots_per_row
    
    # 计算大图总尺寸
    total_width = shots_per_row * cell_total_width
    total_height = num_rows * cell_total_height
    
    # 创建白色背景的最终大图
    final_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    # 粘贴每个单元
    for i, cell in enumerate(cells):
        row = i // shots_per_row
        col = i % shots_per_row
        x = col * cell_total_width
        y = row * cell_total_height
        final_img.paste(cell, (x, y))
    
    # 3. 保存结果
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    final_img.save(output_image_path, quality=90)
    print(f"✅ 可视化图谱已保存至: {output_image_path}")
    return final_img

if __name__ == "__main__":
    # 配置路径
    test_video_path = "./data/test.mp4"
    output_json_path = "./output/structured_video.json"
    output_vis_path = "./output/shot_visualization.jpg"
    
    # 步骤1：全流程结构化
    structured_data = video_structuring(test_video_path, output_json_path)
    
    # 步骤2：可视化
    visualize_shots(test_video_path, structured_data, output_vis_path)


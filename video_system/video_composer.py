"""
视频合成器节点 - 完整版
基于原Coze工作流设计，集成动画效果和转场处理
将音频列表和图片列表合成为具有专业效果的MP4视频
"""

import os
import tempfile
import subprocess
import shutil
import torch
import torchaudio
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import folder_paths
from typing import List, Dict, Any, Tuple, Generator
import json
import math

class VideoComposer:
    """
    视频合成器 - 基础音视频同步合成

    将音频列表和图片batch合成为时间同步的视频
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("*", {"tooltip": "音频列表，每个音频对应一个场景"}),
                "images": ("IMAGE", {"tooltip": "图片batch，每张图片对应一个场景"}),
                "fps": ("INT", {
                    "default": 30,
                    "min": 15,
                    "max": 60,
                    "step": 1,
                    "tooltip": "视频帧率"
                }),
                "width": ("INT", {
                    "default": 720,
                    "min": 480,
                    "max": 1920,
                    "step": 8,
                    "tooltip": "视频宽度"
                }),
                "height": ("INT", {
                    "default": 1280,
                    "min": 720,
                    "max": 2560,
                    "step": 8,
                    "tooltip": "视频高度"
                })
            },
            "optional": {
                "output_format": (["mp4", "avi", "mov"], {
                    "default": "mp4",
                    "tooltip": "输出视频格式"
                }),
                "quality": (["high", "medium", "low"], {
                    "default": "medium",
                    "tooltip": "视频质量"
                }),
                "animation_type": (["coze_zoom", "fade", "slide", "none"], {
                    "default": "coze_zoom",
                    "tooltip": "动画效果类型"
                }),
                "transition_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "转场时长（秒）"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "info")
    FUNCTION = "compose_video"
    CATEGORY = "🎬 Video"
    DESCRIPTION = "将音频列表和图片合成为同步视频"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def compose_video(self, audio_list, images, fps=30, width=720, height=1280,
                     output_format="mp4", quality="medium", animation_type="coze_zoom",
                     transition_duration=0.5):
        """
        合成视频的主函数
        """
        try:
            # 检查输入
            if not isinstance(audio_list, list):
                raise ValueError("audio_list必须是列表类型")

            if len(audio_list) == 0:
                raise ValueError("音频列表不能为空")

            if images.shape[0] != len(audio_list):
                print(f"⚠️ 警告：图片数量({images.shape[0]}) 与音频数量({len(audio_list)}) 不匹配")
                # 调整到最小长度
                min_count = min(images.shape[0], len(audio_list))
                images = images[:min_count]
                audio_list = audio_list[:min_count]

            print(f"🎬 开始视频合成：{len(audio_list)} 个场景，分辨率 {width}x{height}")

            # 1. 分析音频时长
            audio_durations = []
            total_duration = 0

            for i, audio_dict in enumerate(audio_list):
                if isinstance(audio_dict, dict) and "waveform" in audio_dict:
                    waveform = audio_dict["waveform"]
                    sample_rate = audio_dict["sample_rate"]

                    # 计算时长（秒）
                    if len(waveform.shape) == 3:
                        waveform = waveform[0]  # 移除batch维度

                    duration = waveform.shape[1] / sample_rate
                    audio_durations.append(duration)
                    total_duration += duration
                    print(f"🎵 场景 {i+1} 音频时长: {duration:.2f}秒")
                else:
                    raise ValueError(f"音频 {i} 格式不正确，需要包含waveform和sample_rate")

            # 2. 拼接所有音频
            print("🔊 拼接音频...")
            combined_audio_path = self._combine_audio(audio_list)

            # 3. 计算总帧数
            total_frames = sum(int(duration * fps) for duration in audio_durations)
            transition_frames_total = int(transition_duration * fps) * (len(images) - 1) if transition_duration > 0 else 0
            total_frames += transition_frames_total

            print(f"📊 预计生成 {total_frames} 帧 (内存优化模式)")

            # 4. 合成最终视频（使用流式处理）
            print("🎬 开始流式视频合成...")
            output_filename = f"story_video_{self._get_timestamp()}.{output_format}"
            video_path = os.path.join(self.output_dir, output_filename)

            # 创建帧生成器
            frame_generator = self._create_animated_frame_generator(
                images, audio_durations, fps, width, height,
                animation_type, transition_duration
            )

            self._merge_video_audio_streaming(
                frame_generator, combined_audio_path, video_path, fps, quality, total_frames
            )

            # 5. 清理临时文件
            try:
                os.unlink(combined_audio_path)
            except:
                pass

            info = (f"视频合成完成\\n"
                   f"场景数: {len(audio_list)}\\n"
                   f"总时长: {total_duration:.2f}秒\\n"
                   f"分辨率: {width}x{height}\\n"
                   f"帧率: {fps}fps\\n"
                   f"输出: {output_filename}")

            print(f"✅ 视频合成完成: {video_path}")
            return (video_path, info)

        except Exception as e:
            error_msg = f"视频合成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return ("", error_msg)

    def _combine_audio(self, audio_list):
        """拼接所有音频"""
        waveforms = []
        sample_rate = None

        for audio_dict in audio_list:
            waveform = audio_dict["waveform"]
            if len(waveform.shape) == 3:
                waveform = waveform[0]  # 移除batch维度

            waveforms.append(waveform)
            sample_rate = audio_dict["sample_rate"]

        # 拼接音频
        combined_waveform = torch.cat(waveforms, dim=1)

        # 保存为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(temp_file.name, combined_waveform, sample_rate)
        temp_file.close()

        return temp_file.name

    def _create_animated_video_frames(self, images, durations, fps, width, height,
                                    animation_type, transition_duration):
        """
        创建带动画效果的视频帧 - 基于原Coze工作流设计

        支持的动画类型：
        - coze_zoom: 原Coze工作流的缩放动画（奇偶交替方向）
        - fade: 淡入淡出效果
        - slide: 滑动效果
        - none: 无动画
        """
        all_frames = []
        transition_frames = int(transition_duration * fps) if transition_duration > 0 else 0

        for i, (image_tensor, duration) in enumerate(zip(images, durations)):
            # 转换tensor到PIL图片
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # 调整图片尺寸到略大于目标，用于缩放动画
            if animation_type == "coze_zoom":
                # Coze风格：图片稍大一些，用于缩放
                scale_factor = 1.3
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
                pil_image = pil_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            else:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

            # 计算需要的帧数
            total_frames = int(duration * fps)
            content_frames = total_frames - (transition_frames if i < len(images) - 1 else 0)

            print(f"📹 场景 {i+1}: 生成 {total_frames} 帧 ({duration:.2f}秒)")
            print(f"   内容帧: {content_frames}, 转场帧: {transition_frames if i < len(images) - 1 else 0}")

            # 生成动画帧
            scene_frames = self._generate_scene_animation_frames(
                pil_image, content_frames, width, height, animation_type, i
            )
            all_frames.extend(scene_frames)

            # 添加转场效果（除了最后一个场景）
            if i < len(images) - 1 and transition_frames > 0:
                next_image_tensor = images[i + 1]
                next_image_np = (next_image_tensor.cpu().numpy() * 255).astype(np.uint8)
                next_pil_image = Image.fromarray(next_image_np)

                if animation_type == "coze_zoom":
                    next_pil_image = next_pil_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                else:
                    next_pil_image = next_pil_image.resize((width, height), Image.Resampling.LANCZOS)

                transition_frames_list = self._generate_transition_frames(
                    pil_image, next_pil_image, transition_frames, width, height, animation_type
                )
                all_frames.extend(transition_frames_list)

        return all_frames

    def _create_animated_frame_generator(self, images, durations, fps, width, height,
                                       animation_type, transition_duration) -> Generator[np.ndarray, None, None]:
        """
        创建带动画效果的帧生成器（内存优化版本）

        使用Generator模式，每次只生成一帧，避免内存溢出
        """
        transition_frames = int(transition_duration * fps) if transition_duration > 0 else 0

        for i, (image_tensor, duration) in enumerate(zip(images, durations)):
            print(f"📹 处理场景 {i+1}/{len(images)}")

            # 转换tensor到PIL图片
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # 调整图片尺寸到略大于目标，用于缩放动画
            if animation_type == "coze_zoom":
                scale_factor = 1.3
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
                pil_image = pil_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            else:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

            # 计算需要的帧数
            total_frames = int(duration * fps)
            content_frames = total_frames - (transition_frames if i < len(images) - 1 else 0)

            # 生成场景动画帧
            for frame in self._generate_scene_animation_frames_generator(
                pil_image, content_frames, width, height, animation_type, i
            ):
                yield frame

            # 生成转场帧（除了最后一个场景）
            if i < len(images) - 1 and transition_frames > 0:
                next_image_tensor = images[i + 1]
                next_image_np = (next_image_tensor.cpu().numpy() * 255).astype(np.uint8)
                next_pil_image = Image.fromarray(next_image_np)

                if animation_type == "coze_zoom":
                    next_pil_image = next_pil_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                else:
                    next_pil_image = next_pil_image.resize((width, height), Image.Resampling.LANCZOS)

                # 计算转场时的缩放连续性
                current_scene_is_odd = i % 2 == 0
                next_scene_is_odd = (i + 1) % 2 == 0

                # 当前场景结束时的缩放值
                current_end_scale = 1.5 if current_scene_is_odd else 1.0
                # 下一场景开始时的缩放值
                next_start_scale = 1.0 if next_scene_is_odd else 1.5

                print(f"🔄 转场 {i+1}→{i+2}: {current_end_scale:.1f} → {next_start_scale:.1f}")

                for frame in self._generate_transition_frames_generator(
                    pil_image, next_pil_image, transition_frames, width, height,
                    animation_type, current_end_scale, next_start_scale
                ):
                    yield frame

    def _generate_scene_animation_frames_generator(self, pil_image, frame_count, width, height,
                                                 animation_type, scene_index) -> Generator[np.ndarray, None, None]:
        """场景动画帧生成器（内存优化版本）"""

        if animation_type == "coze_zoom":
            # Coze风格缩放动画
            is_odd_scene = scene_index % 2 == 0

            if is_odd_scene:
                start_scale, end_scale = 1.0, 1.5
            else:
                start_scale, end_scale = 1.5, 1.0

            for frame_idx in range(frame_count):
                progress = frame_idx / max(frame_count - 1, 1)
                eased_progress = self._ease_in_out(progress)
                current_scale = start_scale + (end_scale - start_scale) * eased_progress

                # 调试信息：每100帧输出一次当前缩放值
                if frame_idx % 100 == 0:
                    print(f"     帧{frame_idx}: scale={current_scale:.3f} (progress={progress:.3f})")

                frame = self._apply_zoom_effect(pil_image, current_scale, width, height)
                yield frame

        elif animation_type == "fade":
            # 淡入效果
            base_frame = np.array(pil_image.resize((width, height), Image.Resampling.LANCZOS))
            base_frame_bgr = cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)

            for frame_idx in range(frame_count):
                progress = min(frame_idx / (frame_count * 0.2), 1.0)
                alpha = progress
                frame = base_frame_bgr * alpha
                yield frame.astype(np.uint8)

        else:  # none 或其他
            # 静态帧
            base_frame = np.array(pil_image.resize((width, height), Image.Resampling.LANCZOS))
            base_frame_bgr = cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)

            for _ in range(frame_count):
                yield base_frame_bgr

    def _generate_transition_frames_generator(self, current_image, next_image, frame_count,
                                            width, height, animation_type,
                                            current_end_scale=1.0, next_start_scale=1.0) -> Generator[np.ndarray, None, None]:
        """转场帧生成器（内存优化版本，支持缩放连续性）"""

        if animation_type == "coze_zoom":
            # Coze缩放转场：保持缩放连续性的交叉淡化
            for frame_idx in range(frame_count):
                progress = frame_idx / max(frame_count - 1, 1)
                alpha = self._ease_in_out(progress)

                # 当前帧：从结束缩放渐变到1.0
                current_scale = current_end_scale + (1.0 - current_end_scale) * alpha
                current_frame = self._apply_zoom_effect(current_image, current_scale, width, height)

                # 下一帧：从1.0渐变到开始缩放
                next_scale = 1.0 + (next_start_scale - 1.0) * alpha
                next_frame = self._apply_zoom_effect(next_image, next_scale, width, height)

                # 交叉淡化
                current_frame_float = current_frame.astype(np.float32)
                next_frame_float = next_frame.astype(np.float32)

                blended = current_frame_float * (1 - alpha) + next_frame_float * alpha
                yield blended.astype(np.uint8)

        elif animation_type == "fade":
            # 简单淡化转场
            current_frame = np.array(current_image.resize((width, height), Image.Resampling.LANCZOS))
            next_frame = np.array(next_image.resize((width, height), Image.Resampling.LANCZOS))

            current_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            next_bgr = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)

            for frame_idx in range(frame_count):
                progress = frame_idx / max(frame_count - 1, 1)
                alpha = progress
                blended = current_bgr * (1 - alpha) + next_bgr * alpha
                yield blended.astype(np.uint8)

        else:
            # 硬切换转场
            current_frame = np.array(current_image.resize((width, height), Image.Resampling.LANCZOS))
            next_frame = np.array(next_image.resize((width, height), Image.Resampling.LANCZOS))

            current_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            next_bgr = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)

            for frame_idx in range(frame_count):
                progress = frame_idx / max(frame_count - 1, 1)
                if progress < 0.5:
                    yield current_bgr
                else:
                    yield next_bgr

    def _generate_scene_animation_frames(self, pil_image, frame_count, width, height,
                                       animation_type, scene_index):
        """
        为单个场景生成动画帧

        Coze工作流的核心动画：
        - 奇数场景：从1.0缩放到1.5
        - 偶数场景：从1.5缩放到1.0
        """
        frames = []

        if animation_type == "coze_zoom":
            # Coze风格缩放动画
            is_odd_scene = scene_index % 2 == 0  # 0-indexed，所以偶数索引是奇数场景

            if is_odd_scene:
                # 奇数场景：1.0 → 1.5（放大）
                start_scale = 1.0
                end_scale = 1.5
            else:
                # 偶数场景：1.5 → 1.0（缩小）
                start_scale = 1.5
                end_scale = 1.0

            print(f"   Coze缩放: {start_scale:.1f} → {end_scale:.1f}")

            for frame_idx in range(frame_count):
                # 计算当前帧的缩放比例（使用缓动函数）
                progress = frame_idx / max(frame_count - 1, 1)
                # 使用ease-in-out缓动
                eased_progress = self._ease_in_out(progress)
                current_scale = start_scale + (end_scale - start_scale) * eased_progress

                # 应用缩放效果
                frame = self._apply_zoom_effect(pil_image, current_scale, width, height)
                frames.append(frame)

        elif animation_type == "fade":
            # 淡入效果
            base_frame = np.array(pil_image.resize((width, height), Image.Resampling.LANCZOS))
            base_frame_bgr = cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)

            for frame_idx in range(frame_count):
                progress = min(frame_idx / (frame_count * 0.2), 1.0)  # 前20%时间淡入
                alpha = progress
                frame = base_frame_bgr * alpha
                frames.append(frame.astype(np.uint8))

        else:  # none 或其他
            # 静态帧
            base_frame = np.array(pil_image.resize((width, height), Image.Resampling.LANCZOS))
            base_frame_bgr = cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)

            for _ in range(frame_count):
                frames.append(base_frame_bgr)

        return frames

    def _apply_zoom_effect(self, pil_image, scale, target_width, target_height):
        """
        应用Ken Burns缩放效果

        scale = 1.0: 显示完整图片
        scale = 1.5: 放大1.5倍（显示中心2/3区域）
        scale = 2.0: 放大2倍（显示中心1/2区域）

        算法：使用反向思维，将原图看作"虚拟画布"，从中裁剪指定比例
        """
        img_width, img_height = pil_image.size

        # Ken Burns效果：scale越大，看到的图片区域越小（放大效果）
        visible_ratio = 1.0 / scale

        # 计算可见区域尺寸
        visible_width = img_width * visible_ratio
        visible_height = img_height * visible_ratio

        # 转换为整数像素
        crop_width = max(1, int(visible_width))
        crop_height = max(1, int(visible_height))

        # 确保不超出原图边界
        crop_width = min(crop_width, img_width)
        crop_height = min(crop_height, img_height)

        # 居中裁剪
        center_x = img_width // 2
        center_y = img_height // 2

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2
        right = left + crop_width
        bottom = top + crop_height

        # 边界检查
        if left < 0:
            right = right - left
            left = 0
        if right > img_width:
            left = left - (right - img_width)
            right = img_width

        if top < 0:
            bottom = bottom - top
            top = 0
        if bottom > img_height:
            top = top - (bottom - img_height)
            bottom = img_height

        # 裁剪并缩放到目标尺寸
        cropped = pil_image.crop((left, top, right, bottom))
        resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # 转换为OpenCV格式
        frame_np = np.array(resized)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        return frame_bgr

    def _generate_transition_frames(self, current_image, next_image, frame_count,
                                  width, height, animation_type):
        """生成转场帧"""
        frames = []

        current_frame = np.array(current_image.resize((width, height), Image.Resampling.LANCZOS))
        next_frame = np.array(next_image.resize((width, height), Image.Resampling.LANCZOS))

        current_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        next_bgr = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)

        for frame_idx in range(frame_count):
            progress = frame_idx / max(frame_count - 1, 1)

            if animation_type == "fade":
                # 交叉淡化
                alpha = progress
                blended = current_bgr * (1 - alpha) + next_bgr * alpha
                frames.append(blended.astype(np.uint8))
            else:
                # 默认：快速切换（用于其他动画类型）
                if progress < 0.5:
                    frames.append(current_bgr)
                else:
                    frames.append(next_bgr)

        return frames

    def _ease_in_out(self, t):
        """缓动函数：ease-in-out"""
        return t * t * (3.0 - 2.0 * t)

    def _merge_video_audio_streaming(self, frame_generator, audio_path, output_path, fps, quality, total_frames):
        """使用流式处理合成视频，避免内存溢出"""
        temp_video_path = output_path + ".temp.mp4"

        try:
            # 设置视频编码器参数
            if quality == "high":
                crf = 18
            elif quality == "medium":
                crf = 23
            else:  # low
                crf = 28

            print(f"🎬 开始流式写入视频: {total_frames} 帧")

            # 获取第一帧确定视频尺寸
            first_frame = next(frame_generator)
            height, width = first_frame.shape[:2]

            # 使用OpenCV创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise Exception("无法创建视频写入器")

            # 写入第一帧
            out.write(first_frame)
            frames_written = 1

            # 流式写入剩余帧
            for frame in frame_generator:
                out.write(frame)
                frames_written += 1

                # 每1000帧报告进度
                if frames_written % 1000 == 0:
                    progress = (frames_written / total_frames) * 100
                    print(f"📹 写入进度: {frames_written}/{total_frames} ({progress:.1f}%)")

            out.release()
            print(f"✅ 视频写入完成: {frames_written} 帧")

            # 使用ffmpeg添加音频
            self._add_audio_with_ffmpeg(temp_video_path, audio_path, output_path, crf)

        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise e

    def _add_audio_with_ffmpeg(self, video_path, audio_path, output_path, crf):
        """使用ffmpeg添加音频"""
        try:
            # ffmpeg命令 - 使用更好的编码参数
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-crf', str(crf),
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-shortest',  # 以最短的流为准
                output_path
            ]

            print("🔊 添加音频轨道...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 清理临时视频文件
            os.unlink(video_path)
            print("🎵 音频合并成功")

        except subprocess.CalledProcessError as e:
            print(f"⚠️ ffmpeg执行失败: {e.stderr}")
            # 如果ffmpeg失败，使用原视频
            shutil.move(video_path, output_path)
            print("📹 使用无音频视频")

        except FileNotFoundError:
            print("⚠️ ffmpeg未找到，使用无音频视频")
            shutil.move(video_path, output_path)

    def _get_timestamp(self):
        """获取时间戳"""
        import time
        return str(int(time.time()))


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoComposer": VideoComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoComposer": "🎬 视频合成器"
}
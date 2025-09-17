"""
AI视频制作器（原视频合成器-基于视频升级版）
支持两种模式：
1. 传统模式：拼接现有视频片段
2. AI模式：图片通过RunningHub Wan2.2生成视频后拼接
包含完整的音频合成、字体标题、音效库功能
"""

import os
import sys
import tempfile
import subprocess
import shutil
import torch
import torchaudio
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import folder_paths
from typing import List, Dict, Any, Tuple, Generator, Optional
import json
import math
import asyncio
import aiohttp
import ssl
import io
import base64
import time

# 导入音效管理器
try:
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)
    from audio_effects_manager import AudioEffectsManager
    AUDIO_EFFECTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 音效库不可用: {e}")
    AUDIO_EFFECTS_AVAILABLE = False

class AIVideoComposer:
    """
    AI视频制作器
    - 支持图片通过RunningHub Wan2.2 API生成视频
    - 支持传统视频片段拼接
    - 音效库集成
    - 字体标题叠加
    - 音频混合和同步
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = tempfile.gettempdir()

        # 初始化音效管理器
        if AUDIO_EFFECTS_AVAILABLE:
            try:
                self.audio_effects_manager = AudioEffectsManager()
                print("🎵 音效管理器初始化成功")
            except Exception as e:
                print(f"⚠️ 音效管理器初始化失败: {e}")
                self.audio_effects_manager = None
        else:
            self.audio_effects_manager = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 工作模式选择
                "work_mode": (["传统视频拼接", "AI图生视频"], {
                    "default": "AI图生视频",
                    "tooltip": "选择工作模式"
                }),

                "audio_list": ("AUDIO_LIST", {
                    "tooltip": "对应的音频列表"
                }),

                # AI模式专用参数
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "RunningHub API密钥（AI模式必需）"
                }),
                "scene_video_prompts": ("STRING", {
                    "multiline": True,
                    "default": "场景1：美丽的风景慢慢展开\n场景2：人物在画面中移动",
                    "tooltip": "场景图生视频提示词，每行对应一个场景图片"
                }),

                # 基础参数
                "fps": ("INT", {
                    "default": 30,
                    "min": 15,
                    "max": 60,
                    "step": 1,
                    "tooltip": "输出视频帧率"
                }),
                "width": ("INT", {
                    "default": 720,
                    "min": 256,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "输出视频宽度"
                }),
                "height": ("INT", {
                    "default": 1280,
                    "min": 256,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "输出视频高度"
                }),
                "output_format": (["mp4", "avi", "mov"], {
                    "default": "mp4",
                    "tooltip": "输出视频格式"
                }),
                "quality": (["low", "medium", "high", "ultra"], {
                    "default": "medium",
                    "tooltip": "视频质量"
                }),

                # 过渡效果
                "transition_type": (["cut", "fade", "dissolve", "slide"], {
                    "default": "fade",
                    "tooltip": "视频片段间的过渡效果"
                }),
                "transition_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "过渡效果时长（秒）"
                }),

                # 音效系统
                "enable_audio_effects": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用音效增强"
                }),
                "opening_sound_choice": (["无", "自动选择", "史诗开场", "庄重开场", "大气开场", "轻松开场", "神秘开场", "动感开场"], {
                    "default": "自动选择",
                    "tooltip": "开场音效选择"
                }),
                "background_music_choice": (["无", "自动选择", "史诗背景", "庄重背景", "大气背景", "轻松背景", "神秘背景", "动感背景"], {
                    "default": "自动选择",
                    "tooltip": "背景音乐选择"
                }),
                "ambient_sound_choice": (["无", "自动选择", "自然环境", "城市环境", "科技环境", "魔法环境"], {
                    "default": "无",
                    "tooltip": "环境音效选择"
                }),

                # 音量控制
                "background_music_volume": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "背景音乐音量"
                }),
                "opening_sound_volume": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "开场音效音量"
                }),
                "ambient_sound_volume": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "环境音效音量"
                }),
                "voice_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "主音轨音量"
                }),

                # 字体标题系统
                "title_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "标题文字（可选）"
                }),
                "enable_title": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用标题显示"
                }),
                "title_duration": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "标题显示时长（秒）"
                }),
                "title_fontsize": ("INT", {
                    "default": 80,
                    "min": 30,
                    "max": 300,
                    "step": 5,
                    "tooltip": "标题字体大小"
                }),
                "title_color": (["white", "black", "red", "gold", "blue"], {
                    "default": "white",
                    "tooltip": "标题颜色"
                }),
                "title_font": (["自动选择", "Noto无衬线-常规", "Noto无衬线-粗体", "Noto衬线体-粗体", "文泉驿正黑", "超级粗体", "英文粗体"], {
                    "default": "自动选择",
                    "tooltip": "标题字体"
                }),

                # AI模式参数
                "steps": ("INT", {
                    "default": 6,
                    "min": 4,
                    "max": 20,
                    "step": 1,
                    "tooltip": "AI视频生成推理步数"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG缩放系数"
                }),
                "motion_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "运动强度（AI模式）"
                }),
                "max_concurrent": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "tooltip": "AI生成最大并发数"
                }),
            },
            "optional": {
                # 传统模式：现有视频片段
                "video_list": ("VIDEO_LIST", {
                    "tooltip": "传统模式：现有视频片段列表"
                }),

                # AI模式：图片输入
                "images": ("IMAGE", {
                    "tooltip": "AI模式：场景图片列表（不含主角图）"
                }),
                "character_image": ("IMAGE", {
                    "tooltip": "AI模式：主角图片（可选）"
                }),
                "character_video_prompt": ("STRING", {
                    "multiline": True,
                    "default": "主角：人物在画面中自然移动，表情生动",
                    "tooltip": "主角图生视频提示词（单独配置）"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量, 模糊, 变形, 静态",
                    "tooltip": "负面提示词（AI模式）"
                }),
                "enable_character_intro": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用主角开场介绍"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "info")
    FUNCTION = "create_ai_video"
    CATEGORY = "🎥 Shenglin Video System"

    def create_ai_video(self, work_mode, audio_list, api_key, scene_video_prompts, fps=30,
                       width=720, height=1280, output_format="mp4", quality="medium",
                       transition_type="fade", transition_duration=0.5, enable_audio_effects=True,
                       opening_sound_choice="自动选择", background_music_choice="自动选择",
                       ambient_sound_choice="无", background_music_volume=0.3,
                       opening_sound_volume=0.8, ambient_sound_volume=0.5, voice_volume=1.0,
                       title_text="", enable_title=False, title_duration=3.0,
                       title_fontsize=80, title_color="white", title_font="自动选择",
                       steps=6, cfg_scale=7.5, motion_strength=0.8, max_concurrent=2,
                       video_list=None, images=None, character_image=None,
                       character_video_prompt="主角：人物在画面中自然移动，表情生动",
                       negative_prompt="低质量, 模糊, 变形, 静态", enable_character_intro=True):
        """
        AI视频制作器主函数
        支持传统视频拼接和AI图生视频两种模式
        """
        try:
            print(f"🎬 开始{work_mode}...")

            if not audio_list or len(audio_list) == 0:
                return ("", "错误: 音频列表为空")

            # 根据工作模式选择处理流程
            if work_mode == "AI图生视频":
                return self._handle_ai_mode(
                    audio_list, api_key, scene_video_prompts, fps, width, height,
                    output_format, quality, transition_type, transition_duration,
                    enable_audio_effects, opening_sound_choice, background_music_choice,
                    ambient_sound_choice, background_music_volume, opening_sound_volume,
                    ambient_sound_volume, voice_volume, title_text, enable_title,
                    title_duration, title_fontsize, title_color, title_font,
                    steps, cfg_scale, motion_strength, max_concurrent,
                    images, character_image, character_video_prompt,
                    negative_prompt, enable_character_intro
                )
            else:
                # 传统模式：基于现有视频片段
                if not video_list or len(video_list) == 0:
                    return ("", "错误: 传统模式需要提供视频片段列表")

                return self._handle_traditional_mode(
                    video_list, audio_list, fps, width, height, output_format, quality,
                    transition_type, transition_duration, enable_audio_effects,
                    opening_sound_choice, background_music_choice, ambient_sound_choice,
                    background_music_volume, opening_sound_volume, ambient_sound_volume,
                    voice_volume, title_text, enable_title, title_duration,
                    title_fontsize, title_color, title_font
                )

        except Exception as e:
            error_msg = f"AI视频制作失败: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", error_msg)

    def _handle_traditional_mode(self, video_list, audio_list, fps, width, height,
                               output_format, quality, transition_type, transition_duration,
                               enable_audio_effects, opening_sound_choice, background_music_choice,
                               ambient_sound_choice, background_music_volume, opening_sound_volume,
                               ambient_sound_volume, voice_volume, title_text, enable_title,
                               title_duration, title_fontsize, title_color, title_font):
        """传统模式：基于现有视频片段的合成"""
        try:
            print(f"📋 输入信息: {len(video_list)}个视频片段, {len(audio_list)}个音频段")

            # 计算音频时长
            audio_durations = []
            for i, audio_dict in enumerate(audio_list):
                waveform = audio_dict["waveform"]
                if len(waveform.shape) == 3:
                    waveform = waveform[0]
                sample_rate = audio_dict["sample_rate"]
                duration = waveform.shape[1] / sample_rate
                audio_durations.append(duration)
                print(f"🎵 音频{i+1}: {duration:.2f}秒")

            total_duration = sum(audio_durations)

            # 生成输出文件名
            import time
            timestamp = str(int(time.time()))
            output_filename = f"video_from_videos_{timestamp}.{output_format}"
            video_path = os.path.join(self.output_dir, output_filename)

            # 第一步：拼接视频片段
            print("🔗 拼接视频片段...")
            concatenated_video = self._concatenate_videos(video_list, transition_type, transition_duration, fps)

            if not concatenated_video:
                return ("", "错误: 视频拼接失败")

            # 第二步：调整视频尺寸和时长
            print("📐 调整视频尺寸和同步...")
            resized_video = self._resize_and_sync_video(concatenated_video, width, height, total_duration, fps)

            # 第三步：合成音频
            print("🎵 合成音频轨道...")
            combined_audio_path = self._compose_enhanced_audio(
                audio_list, audio_durations, enable_audio_effects,
                opening_sound_choice, background_music_choice, ambient_sound_choice,
                background_music_volume, opening_sound_volume, ambient_sound_volume, voice_volume
            )

            # 第四步：合成最终视频
            print("🎬 合成最终视频...")
            final_video = self._combine_video_audio_with_title(
                resized_video, combined_audio_path, video_path, quality,
                title_text, enable_title, title_duration, title_fontsize, title_color, title_font
            )

            # 清理临时文件
            self._cleanup_temp_files([concatenated_video, resized_video, combined_audio_path])

            if not final_video:
                return ("", "错误: 最终视频合成失败")

            # 生成信息
            info = (f"基于视频的视频合成完成\\n"
                   f"视频片段: {len(video_list)}个\\n"
                   f"音频段数: {len(audio_list)}\\n"
                   f"总时长: {total_duration:.2f}秒\\n"
                   f"分辨率: {width}x{height}\\n"
                   f"帧率: {fps}fps\\n"
                   f"过渡效果: {transition_type}\\n"
                   f"输出: {output_filename}")

            print(f"✅ 基于视频的合成完成: {video_path}")
            return (video_path, info)

        except Exception as e:
            error_msg = f"基于视频的合成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return ("", error_msg)

    def _handle_ai_mode(self, audio_list, api_key, scene_video_prompts, fps, width, height,
                       output_format, quality, transition_type, transition_duration,
                       enable_audio_effects, opening_sound_choice, background_music_choice,
                       ambient_sound_choice, background_music_volume, opening_sound_volume,
                       ambient_sound_volume, voice_volume, title_text, enable_title,
                       title_duration, title_fontsize, title_color, title_font,
                       steps, cfg_scale, motion_strength, max_concurrent,
                       images, character_image, character_video_prompt,
                       negative_prompt, enable_character_intro):
        """AI模式：图片通过RunningHub Wan2.2生成视频后拼接"""
        try:
            if not api_key.strip():
                return ("", "错误: AI模式需要提供RunningHub API密钥")

            # 解析场景视频提示词
            scene_prompts = [line.strip() for line in scene_video_prompts.strip().split('\n') if line.strip()]
            if not scene_prompts:
                return ("", "错误: 场景视频提示词不能为空")

            print(f"🎬 AI模式: {len(scene_prompts)}个场景提示词, {len(audio_list)}个音频")

            # 构建输入图片列表（主角图+场景图）
            all_images = []
            all_prompts = []

            # 处理主角图（如果有）
            if character_image is not None and enable_character_intro:
                if len(character_image.shape) == 4:
                    # 批量图片，取第一张作为主角
                    char_img = character_image[0]
                else:
                    char_img = character_image
                all_images.append(char_img)
                all_prompts.append(character_video_prompt.strip())
                print("👤 添加主角图片")

            # 处理场景图片
            if images is not None:
                if len(images.shape) == 4:
                    # 批量图片
                    scene_count = min(len(scene_prompts), images.shape[0])
                    for i in range(scene_count):
                        all_images.append(images[i])
                        prompt_idx = min(i, len(scene_prompts) - 1)
                        all_prompts.append(scene_prompts[prompt_idx])
                    print(f"🖼️ 添加 {scene_count} 张场景图片")
                else:
                    # 单张场景图片
                    all_images.append(images)
                    all_prompts.append(scene_prompts[0] if scene_prompts else "场景视频")
                    print("🖼️ 添加 1 张场景图片")

            if not all_images:
                return ("", "错误: 未提供任何图片（请提供场景图片或主角图片）")

            # 确保音频和图片数量匹配
            if len(all_images) != len(audio_list):
                return ("", f"错误: 图片数量({len(all_images)})与音频数量({len(audio_list)})不匹配")

            # 调用RunningHub Wan2.2 API生成视频
            print("🚀 开始调用RunningHub Wan2.2 API生成视频...")
            video_paths = self._generate_videos_with_runninghub(
                all_images, all_prompts, audio_list, api_key, steps, cfg_scale,
                motion_strength, max_concurrent, negative_prompt
            )

            if not video_paths or not any(video_paths):
                return ("", "错误: RunningHub API视频生成失败")

            print(f"✅ 成功生成 {len([p for p in video_paths if p])} 个视频片段")

            # 使用生成的视频进行后续合成（调用传统模式逻辑）
            return self._handle_traditional_mode(
                video_paths, audio_list, fps, width, height, output_format, quality,
                transition_type, transition_duration, enable_audio_effects,
                opening_sound_choice, background_music_choice, ambient_sound_choice,
                background_music_volume, opening_sound_volume, ambient_sound_volume,
                voice_volume, title_text, enable_title, title_duration,
                title_fontsize, title_color, title_font
            )

        except Exception as e:
            error_msg = f"AI模式处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", error_msg)

    def _concatenate_videos(self, video_list, transition_type, transition_duration, fps):
        """拼接视频片段"""
        try:
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video.close()

            if transition_type == "cut":
                # 直接拼接，无过渡
                concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                for video_path in video_list:
                    concat_list.write(f"file '{video_path}'\\n")
                concat_list.close()

                cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list.name,
                       '-c', 'copy', temp_video.name]

                os.unlink(concat_list.name)
            else:
                # 带过渡效果的拼接
                cmd = ['ffmpeg', '-y']

                # 添加所有输入视频
                for video_path in video_list:
                    cmd.extend(['-i', video_path])

                # 构建过渡滤镜
                if transition_type == "fade":
                    filter_complex = self._build_fade_filter(len(video_list), transition_duration)
                elif transition_type == "dissolve":
                    filter_complex = self._build_dissolve_filter(len(video_list), transition_duration)
                elif transition_type == "slide":
                    filter_complex = self._build_slide_filter(len(video_list), transition_duration)
                else:
                    filter_complex = self._build_fade_filter(len(video_list), transition_duration)

                cmd.extend(['-filter_complex', filter_complex, temp_video.name])

            subprocess.run(cmd, check=True, capture_output=True)
            return temp_video.name

        except Exception as e:
            print(f"❌ 视频拼接失败: {e}")
            return None

    def _build_fade_filter(self, video_count, transition_duration):
        """构建淡入淡出过渡滤镜"""
        filter_parts = []

        for i in range(video_count):
            if i == 0:
                # 第一个视频
                filter_parts.append(f"[{i}:v]fade=t=out:st=0:d={transition_duration}[v{i}]")
            elif i == video_count - 1:
                # 最后一个视频
                filter_parts.append(f"[{i}:v]fade=t=in:st=0:d={transition_duration}[v{i}]")
            else:
                # 中间视频
                filter_parts.append(f"[{i}:v]fade=t=in:st=0:d={transition_duration},fade=t=out:st=0:d={transition_duration}[v{i}]")

        # 拼接所有视频
        concat_inputs = "".join([f"[v{i}]" for i in range(video_count)])
        filter_parts.append(f"{concat_inputs}concat=n={video_count}:v=1:a=0[outv]")

        return ";".join(filter_parts)

    def _build_dissolve_filter(self, video_count, transition_duration):
        """构建溶解过渡滤镜"""
        # 简化版本，使用fade作为溶解效果
        return self._build_fade_filter(video_count, transition_duration)

    def _build_slide_filter(self, video_count, transition_duration):
        """构建滑动过渡滤镜"""
        # 简化版本，使用fade作为滑动效果
        return self._build_fade_filter(video_count, transition_duration)

    def _resize_and_sync_video(self, video_path, width, height, target_duration, fps):
        """调整视频尺寸和同步时长"""
        try:
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video.close()

            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
                '-t', str(target_duration),
                '-r', str(fps),
                '-c:v', 'libx264',
                '-preset', 'medium',
                temp_video.name
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            return temp_video.name

        except Exception as e:
            print(f"❌ 视频调整失败: {e}")
            return None

    def _compose_enhanced_audio(self, audio_list, durations, enable_audio_effects,
                              opening_sound_choice, background_music_choice, ambient_sound_choice,
                              background_music_volume, opening_sound_volume, ambient_sound_volume, voice_volume):
        """增强音频合成（复用现有逻辑）"""
        try:
            # 合成主音轨
            main_audio_path = self._combine_audio(audio_list)

            if not enable_audio_effects or not self.audio_effects_manager:
                return main_audio_path

            # 应用音效增强
            enhanced_audio = self.audio_effects_manager.enhance_audio_with_effects(
                main_audio_path, durations,
                opening_sound=opening_sound_choice,
                background_music=background_music_choice,
                ambient_sound=ambient_sound_choice,
                bg_volume=background_music_volume,
                opening_volume=opening_sound_volume,
                ambient_volume=ambient_sound_volume,
                voice_volume=voice_volume
            )

            return enhanced_audio if enhanced_audio else main_audio_path

        except Exception as e:
            print(f"❌ 音频合成失败: {e}")
            return self._combine_audio(audio_list)

    def _combine_audio(self, audio_list):
        """拼接音频"""
        waveforms = []
        sample_rate = None

        for audio_dict in audio_list:
            waveform = audio_dict["waveform"]
            if len(waveform.shape) == 3:
                waveform = waveform[0]
            waveforms.append(waveform)
            sample_rate = audio_dict["sample_rate"]

        combined_waveform = torch.cat(waveforms, dim=1)

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(temp_file.name, combined_waveform, sample_rate)
        temp_file.close()

        return temp_file.name

    def _combine_video_audio_with_title(self, video_path, audio_path, output_path, quality,
                                      title_text, enable_title, title_duration, title_fontsize,
                                      title_color, title_font):
        """合成视频、音频和标题"""
        try:
            # 质量设置
            quality_settings = {
                "low": ["-crf", "28", "-preset", "fast"],
                "medium": ["-crf", "23", "-preset", "medium"],
                "high": ["-crf", "18", "-preset", "slow"],
                "ultra": ["-crf", "15", "-preset", "slower"]
            }

            cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path]

            # 添加文字滤镜（如果启用标题）
            if enable_title and title_text.strip():
                font_path = self._get_font_path(title_font)
                title_text_escaped = title_text.replace(":", "\\:")

                if font_path:
                    text_filter = f"drawtext=text='{title_text_escaped}':fontfile='{font_path}':fontsize={title_fontsize}:fontcolor={title_color}:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,{title_duration})'"
                else:
                    text_filter = f"drawtext=text='{title_text_escaped}':fontsize={title_fontsize}:fontcolor={title_color}:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,{title_duration})'"

                cmd.extend(['-vf', text_filter])
                print(f"📝 添加标题: '{title_text}' (显示{title_duration}秒)")

            # 添加质量设置
            cmd.extend(quality_settings[quality])
            cmd.extend(['-c:a', 'aac', '-b:a', '192k', output_path])

            subprocess.run(cmd, check=True, capture_output=True)
            return True

        except Exception as e:
            print(f"❌ 最终合成失败: {e}")
            return False

    def _get_font_path(self, font_name):
        """获取字体路径（复用现有逻辑）"""
        try:
            # 导入字体映射逻辑
            font_name_mapping = {
                "自动选择": "auto",
                "Noto无衬线-常规": "noto_cjk_regular",
                "Noto无衬线-粗体": "noto_cjk_bold",
                "Noto衬线体-粗体": "noto_serif_cjk_bold",
                "文泉驿正黑": "wqy_zenhei",
                "超级粗体": "nimbus_sans_bold",
                "英文粗体": "noto_serif_bold"
            }

            if font_name in font_name_mapping:
                font_name = font_name_mapping[font_name]

            # 获取内置字体包路径
            bundled_fonts_dir = self._get_bundled_fonts_dir()
            if not bundled_fonts_dir:
                return None

            if font_name == "auto":
                # 自动选择（粗体优先）
                priority_fonts = [
                    os.path.join(bundled_fonts_dir, "NotoSansCJK-Bold.ttc"),
                    os.path.join(bundled_fonts_dir, "NotoSerifCJK-Bold.ttc"),
                    os.path.join(bundled_fonts_dir, "wqy-zenhei.ttc"),
                    os.path.join(bundled_fonts_dir, "NotoSansCJK-Regular.ttc"),
                ]
                for font_path in priority_fonts:
                    if os.path.exists(font_path):
                        return font_path

            # 字体映射
            font_mapping = {
                "noto_cjk_regular": os.path.join(bundled_fonts_dir, "NotoSansCJK-Regular.ttc"),
                "noto_cjk_bold": os.path.join(bundled_fonts_dir, "NotoSansCJK-Bold.ttc"),
                "noto_serif_cjk_bold": os.path.join(bundled_fonts_dir, "NotoSerifCJK-Bold.ttc"),
                "wqy_zenhei": os.path.join(bundled_fonts_dir, "wqy-zenhei.ttc"),
                "nimbus_sans_bold": os.path.join(bundled_fonts_dir, "NimbusSans-Bold.otf"),
                "noto_serif_bold": os.path.join(bundled_fonts_dir, "NotoSerif-Bold.ttf"),
            }

            if font_name in font_mapping:
                font_path = font_mapping[font_name]
                if os.path.exists(font_path):
                    return font_path

            return None

        except Exception as e:
            print(f"⚠️ 字体获取失败: {e}")
            return None

    def _get_bundled_fonts_dir(self):
        """获取内置字体包目录"""
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fonts_dir = os.path.join(current_dir, "fonts")
            return fonts_dir if os.path.exists(fonts_dir) else None
        except:
            return None

    def _cleanup_temp_files(self, file_paths):
        """清理临时文件"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass

    def _generate_videos_with_runninghub(self, images, prompts, audio_list, api_key,
                                       steps, cfg_scale, motion_strength, max_concurrent, negative_prompt):
        """使用RunningHub Wan2.2 API生成视频"""
        try:
            # 获取音频时长列表
            audio_durations = self._get_audio_durations(audio_list)

            # 转换图片格式
            image_list = []
            for img_tensor in images:
                if isinstance(img_tensor, torch.Tensor):
                    img_array = img_tensor.cpu().numpy()
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    image_list.append(Image.fromarray(img_array))
                else:
                    image_list.append(img_tensor)

            print(f"🎬 准备生成 {len(image_list)} 个视频片段")

            # 异步调用API
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._batch_generate_videos_async(
                        image_list, prompts, audio_durations, api_key, steps, cfg_scale,
                        motion_strength, max_concurrent, negative_prompt
                    ))
                    results = future.result()
            except RuntimeError as e:
                if "There is no current event loop" in str(e):
                    results = asyncio.run(
                        self._batch_generate_videos_async(
                            image_list, prompts, audio_durations, api_key, steps, cfg_scale,
                            motion_strength, max_concurrent, negative_prompt
                        )
                    )
                else:
                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(
                        self._batch_generate_videos_async(
                            image_list, prompts, audio_durations, api_key, steps, cfg_scale,
                            motion_strength, max_concurrent, negative_prompt
                        )
                    )

            # 提取视频路径
            video_paths = []
            for result in results:
                if result and result.get('success'):
                    video_paths.append(result.get('video_path', ''))
                else:
                    video_paths.append('')

            return video_paths

        except Exception as e:
            print(f"❌ RunningHub API调用失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _get_audio_durations(self, audio_list):
        """获取音频时长列表"""
        durations = []
        for audio_dict in audio_list:
            waveform = audio_dict["waveform"]
            if len(waveform.shape) == 3:
                waveform = waveform[0]
            sample_rate = audio_dict["sample_rate"]
            duration = waveform.shape[1] / sample_rate
            durations.append(float(duration))
        return durations

    async def _batch_generate_videos_async(self, images, prompts, audio_durations, api_key,
                                         steps, cfg_scale, motion_strength, max_concurrent, negative_prompt):
        """异步批量生成视频"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        fps = 16  # Wan2.2固定帧率

        for i, (image, prompt, duration) in enumerate(zip(images, prompts, audio_durations)):
            frames = max(25, min(200, int(duration * fps)))  # 根据音频时长计算帧数

            task = asyncio.create_task(
                self._generate_single_video_async(
                    semaphore, i, image, prompt, frames, api_key, steps,
                    cfg_scale, motion_strength, negative_prompt
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _generate_single_video_async(self, semaphore, index, image, prompt, frames,
                                         api_key, steps, cfg_scale, motion_strength, negative_prompt):
        """异步生成单个视频"""
        async with semaphore:
            try:
                print(f"🎬 生成视频 {index+1}: {frames}帧, {prompt[:30]}...")

                # 转换图片为base64
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # 生成随机种子
                seed = int(time.time() * 1000) % 2147483647

                # 构建API请求参数
                workflow_id = "1968308523518046210"
                node_list = [
                    {
                        "nodeId": 1,
                        "type": "input_image",
                        "data": {
                            "image": image_b64,
                            "format": "png"
                        }
                    },
                    {
                        "nodeId": 2,
                        "type": "text_prompt",
                        "data": {
                            "text": prompt
                        }
                    },
                    {
                        "nodeId": 3,
                        "type": "video_params",
                        "data": {
                            "frames": frames,
                            "steps": steps,
                            "cfg_scale": cfg_scale,
                            "seed": seed,
                            "motion_strength": motion_strength
                        }
                    }
                ]

                if negative_prompt.strip():
                    node_list.append({
                        "nodeId": 4,
                        "type": "negative_prompt",
                        "data": {
                            "text": negative_prompt.strip()
                        }
                    })

                payload = {
                    "apiKey": api_key,
                    "workflowId": workflow_id,
                    "nodeInfoList": node_list
                }

                # 发送API请求
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                connector = aiohttp.TCPConnector(ssl=ssl_context)
                timeout = aiohttp.ClientTimeout(total=1200)

                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    async with session.post(
                        'https://api.runninghub.cn/api/v1/workflows/run',
                        headers={
                            'Content-Type': 'application/json',
                            'User-Agent': 'ComfyUI-AIVideoComposer/1.0'
                        },
                        json=payload
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"API请求失败: {response.status} - {error_text}")

                        result = await response.json()
                        if not result.get('success', False):
                            error_msg = result.get('message', '未知错误')
                            raise Exception(f"视频生成失败: {error_msg}")

                        # 获取视频URL并下载
                        video_data = result.get('data', {})
                        video_url = video_data.get('videoUrl') or video_data.get('output_video')

                        if not video_url:
                            raise Exception("视频URL为空")

                        video_path = await self._download_video_async(session, video_url, prompt, index)

                        print(f"✅ 视频 {index+1} 生成成功: {video_path}")
                        return {
                            "success": True,
                            "video_path": video_path,
                            "frames": frames,
                            "prompt": prompt
                        }

            except Exception as e:
                print(f"❌ 视频 {index+1} 生成失败: {e}")
                return {"success": False, "error": str(e)}

    async def _download_video_async(self, session, video_url, prompt, index):
        """异步下载视频"""
        try:
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_'))[:30]
            timestamp = int(time.time())
            filename = f"ai_video_{index+1}_{timestamp}_{safe_prompt[:10]}.mp4"
            video_path = os.path.join(self.output_dir, filename)

            async with session.get(video_url) as response:
                if response.status == 200:
                    with open(video_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return video_path
                else:
                    raise Exception(f"下载失败: {response.status}")

        except Exception as e:
            print(f"❌ 视频下载失败: {e}")
            return video_url  # 返回原始URL作为备选

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AIVideoComposer": AIVideoComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIVideoComposer": "🎬 AI视频制作器"
}
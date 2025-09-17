"""
视频合成器-基于图片
将静态图片转换为动态视频，支持图像动画、音效库、字体标题等全面功能
适用于从图片生成视频的场景
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

# 导入音效管理器（从当前节点目录）
try:
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)
    from audio_effects_manager import AudioEffectsManager
    AUDIO_EFFECTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 音效库不可用: {e}")
    AUDIO_EFFECTS_AVAILABLE = False

class EnhancedVideoComposerV2:
    """
    增强版视频合成器V2 - 支持音效选择的3轨音频系统

    新增功能：
    1. 音效文件选择
    2. 音效风格/标签筛选
    3. 自定义音效导入
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用音效选项
        audio_effects_options = cls._get_audio_effects_options()

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
                }),

                # 🎵 音效选择参数
                "enable_audio_effects": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用音效库（背景音乐+开场音效）"
                }),
                "opening_sound_choice": (audio_effects_options["opening"], {
                    "default": audio_effects_options["opening"][0] if audio_effects_options["opening"] else "无",
                    "tooltip": "选择开场音效"
                }),
                "background_music_choice": (audio_effects_options["background"], {
                    "default": audio_effects_options["background"][0] if audio_effects_options["background"] else "无",
                    "tooltip": "选择背景音乐"
                }),
                "ambient_sound_choice": (audio_effects_options["ambient"], {
                    "default": audio_effects_options["ambient"][0] if audio_effects_options["ambient"] else "无",
                    "tooltip": "选择环境音效"
                }),

                # 🎚️ 音量控制
                "background_music_volume": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "背景音乐音量（0.3=30%）"
                }),
                "opening_sound_volume": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "开场音效音量（0.8=80%）"
                }),
                "ambient_sound_volume": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "环境音效音量（0.5=50%）"
                }),
                "voice_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "主音轨（语音）音量"
                }),

                # 📋 音效风格筛选
                "audio_style_filter": (["全部", "史诗", "历史", "庄重", "大气", "轻松", "神秘", "动感"], {
                    "default": "全部",
                    "tooltip": "按风格筛选音效"
                }),

                # 主角图相关参数
                "character_image": ("IMAGE", {
                    "tooltip": "主角图片，用于首帧特效（可选）"
                }),
                "enable_character_intro": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用主角开场动画"
                }),
                "char_intro_scale_start": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "主角图开始缩放比例"
                }),
                "char_intro_scale_mid": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "主角图中间缩放比例"
                }),
                "char_intro_scale_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "主角图结束缩放比例"
                }),
                "char_intro_mid_timing": ("FLOAT", {
                    "default": 0.533,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "主角图中间关键帧时间点（秒）"
                }),
                # 标题显示参数
                "title_text": ("STRING", {
                    "default": "",
                    "tooltip": "视频标题文字（2字主题）"
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
                "title_font": (["自动选择", "Noto无衬线-常规", "Noto无衬线-粗体", "Noto衬线体", "文泉驿正黑", "Droid黑体", "Arial风格", "Helvetica风格", "Times风格"], {
                    "default": "自动选择",
                    "tooltip": "标题字体选择（支持中英文）"
                })
            }
        }

    @classmethod
    def _get_audio_effects_options(cls):
        """获取可用音效选项"""
        options = {
            "opening": ["无", "自动选择"],
            "background": ["无", "自动选择"],
            "ambient": ["无", "自动选择"]
        }

        if not AUDIO_EFFECTS_AVAILABLE:
            return options

        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, current_dir)
            from audio_effects_manager import AudioEffectsManager

            manager = AudioEffectsManager()

            # 获取开场音效列表
            opening_files = manager.list_files("opening")
            for file_info in opening_files:
                name = file_info.get("name", file_info.get("filename", "未知"))
                options["opening"].append(name)

            # 获取背景音乐列表
            bg_files = manager.list_files("background")
            for file_info in bg_files:
                name = file_info.get("name", file_info.get("filename", "未知"))
                options["background"].append(name)

            # 获取环境音效列表
            ambient_files = manager.list_files("ambient")
            for file_info in ambient_files:
                name = file_info.get("name", file_info.get("filename", "未知"))
                options["ambient"].append(name)

        except Exception as e:
            print(f"获取音效选项失败: {e}")

        return options

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "info")
    FUNCTION = "compose_video_with_selectable_effects"
    CATEGORY = "🔥 Shenglin/视频处理"
    DESCRIPTION = "增强版视频合成器V2，支持音效选择的3轨音频系统"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

        # 初始化音效管理器
        if AUDIO_EFFECTS_AVAILABLE:
            try:
                self.audio_effects_manager = AudioEffectsManager()
                print("✅ 音效库V2初始化成功")
            except Exception as e:
                print(f"❌ 音效库V2初始化失败: {e}")
                self.audio_effects_manager = None
        else:
            self.audio_effects_manager = None

    def compose_video_with_selectable_effects(self, audio_list, images, fps=30, width=720, height=1280,
                                            output_format="mp4", quality="medium", animation_type="coze_zoom",
                                            transition_duration=0.5, enable_audio_effects=True,
                                            opening_sound_choice="自动选择", background_music_choice="自动选择",
                                            ambient_sound_choice="无", background_music_volume=0.3,
                                            opening_sound_volume=0.8, ambient_sound_volume=0.5, voice_volume=1.0,
                                            audio_style_filter="全部", character_image=None, enable_character_intro=True,
                                            char_intro_scale_start=2.0, char_intro_scale_mid=1.2,
                                            char_intro_scale_end=1.0, char_intro_mid_timing=0.533,
                                            title_text="", enable_title=False, title_duration=3.0,
                                            title_fontsize=80, title_color="white", title_font="自动选择"):
        """
        增强版视频合成V2 - 支持音效选择
        """
        try:
            # 基础检查
            if not isinstance(audio_list, list) or len(audio_list) == 0:
                raise ValueError("音频列表不能为空")

            if images.shape[0] != len(audio_list):
                print(f"⚠️ 警告：图片数量({images.shape[0]}) 与音频数量({len(audio_list)}) 不匹配")
                min_count = min(images.shape[0], len(audio_list))
                images = images[:min_count]
                audio_list = audio_list[:min_count]

            print(f"🎬 开始增强版视频合成V2：{len(audio_list)} 个场景")
            print(f"🎵 音效选择：开场={opening_sound_choice}, 背景={background_music_choice}, 环境={ambient_sound_choice}")

            # 1. 分析音频时长
            audio_durations = []
            total_duration = 0

            for i, audio_dict in enumerate(audio_list):
                if isinstance(audio_dict, dict) and "waveform" in audio_dict:
                    waveform = audio_dict["waveform"]
                    sample_rate = audio_dict["sample_rate"]

                    if len(waveform.shape) == 3:
                        waveform = waveform[0]

                    duration = waveform.shape[1] / sample_rate
                    audio_durations.append(duration)
                    total_duration += duration
                    print(f"🎵 场景 {i+1} 音频时长: {duration:.2f}秒")
                else:
                    raise ValueError(f"音频 {i} 格式不正确")

            # 2. 合成选择性音频系统
            print("🔊 开始选择性音频合成...")
            if enable_audio_effects and self.audio_effects_manager:
                combined_audio_path = self._combine_audio_with_selectable_effects(
                    audio_list, total_duration,
                    opening_sound_choice, background_music_choice, ambient_sound_choice,
                    background_music_volume, opening_sound_volume, ambient_sound_volume, voice_volume,
                    audio_style_filter
                )
            else:
                print("📢 使用单轨音频模式")
                combined_audio_path = self._combine_audio_simple(audio_list, voice_volume)

            # 3. 视频合成（复用原有逻辑）
            total_frames = sum(int(duration * fps) for duration in audio_durations)
            transition_frames_total = int(transition_duration * fps) * (len(images) - 1) if transition_duration > 0 else 0
            total_frames += transition_frames_total

            print("🎬 开始视频合成...")
            output_filename = f"story_video_v2_{self._get_timestamp()}.{output_format}"
            video_path = os.path.join(self.output_dir, output_filename)

            # 导入原始合成逻辑
            from .video_composer import VideoComposer
            original_composer = VideoComposer()

            frame_generator = original_composer._create_animated_frame_generator(
                images, audio_durations, fps, width, height,
                animation_type, transition_duration, character_image, enable_character_intro,
                char_intro_scale_start, char_intro_scale_mid, char_intro_scale_end, char_intro_mid_timing
            )

            # 构建标题配置
            title_config = {
                'enable_title': enable_title and title_text.strip(),
                'title_text': title_text.strip(),
                'fontsize': title_fontsize,
                'color': title_color,
                'duration': title_duration,
                'font': title_font
            } if enable_title and title_text.strip() else None

            original_composer._merge_video_audio_streaming(
                frame_generator, combined_audio_path, video_path, fps, quality, total_frames, title_config
            )

            # 4. 清理临时文件
            try:
                os.unlink(combined_audio_path)
            except:
                pass

            # 5. 生成详细信息报告
            effect_info = self._generate_effects_info(
                enable_audio_effects, opening_sound_choice, background_music_choice,
                ambient_sound_choice, audio_style_filter
            )

            title_info = f"'{title_text}' ({title_duration}s)" if enable_title and title_text.strip() else "未启用"

            info = f"""🎬 增强版视频合成V2完成
📁 输出路径: {video_path}
📊 视频规格: {width}x{height}@{fps}fps
🎵 音效配置: {effect_info}
👤 主角动画: {'已启用' if enable_character_intro else '未启用'}
📝 视频标题: {title_info}
📈 总时长: {total_duration:.2f}秒
🎞️ 总帧数: {total_frames}"""

            print("✅ 增强版视频合成V2完成！")
            return (video_path, info)

        except Exception as e:
            error_msg = f"❌ 增强版视频合成V2失败: {str(e)}"
            print(error_msg)
            return ("", error_msg)

    def _combine_audio_with_selectable_effects(self, audio_list, total_duration,
                                             opening_choice, background_choice, ambient_choice,
                                             bg_volume, opening_volume, ambient_volume, voice_volume,
                                             style_filter):
        """
        选择性音频合成：根据用户选择合成音轨
        """
        try:
            # 1. 合成主音轨（语音）
            voice_waveforms = []
            sample_rate = None

            for audio_dict in audio_list:
                waveform = audio_dict["waveform"]
                if len(waveform.shape) == 3:
                    waveform = waveform[0]
                voice_waveforms.append(waveform)
                sample_rate = audio_dict["sample_rate"]

            voice_combined = torch.cat(voice_waveforms, dim=1) * voice_volume
            print(f"🎤 主音轨：{voice_combined.shape[1]/sample_rate:.2f}秒，音量{voice_volume*100:.0f}%")

            # 2. 处理背景音乐
            bg_waveform = self._load_selected_audio("background", background_choice,
                                                  total_duration, sample_rate, bg_volume, style_filter)

            # 3. 处理开场音效
            opening_track = self._load_selected_audio("opening", opening_choice,
                                                    voice_combined.shape[1]/sample_rate, sample_rate,
                                                    opening_volume, style_filter, is_opening=True)

            # 4. 处理环境音效
            ambient_track = self._load_selected_audio("ambient", ambient_choice,
                                                    total_duration, sample_rate, ambient_volume, style_filter)

            # 5. 混合所有音轨
            min_length = min(voice_combined.shape[1], bg_waveform.shape[1],
                           opening_track.shape[1], ambient_track.shape[1])

            voice_combined = voice_combined[:, :min_length]
            bg_waveform = bg_waveform[:, :min_length]
            opening_track = opening_track[:, :min_length]
            ambient_track = ambient_track[:, :min_length]

            # 混合音频
            final_audio = voice_combined + bg_waveform + opening_track + ambient_track

            # 防止音频剪切
            max_val = torch.max(torch.abs(final_audio))
            if max_val > 1.0:
                final_audio = final_audio / max_val * 0.95
                print(f"🔧 音频压缩：峰值从{max_val:.2f}压缩到0.95")

            # 保存混合音频
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_file.name, final_audio, sample_rate)
            temp_file.close()

            print(f"✅ 选择性音频合成完成：{final_audio.shape[1]/sample_rate:.2f}秒")
            return temp_file.name

        except Exception as e:
            print(f"❌ 选择性音频合成失败: {e}")
            return self._combine_audio_simple(audio_list, voice_volume)

    def _load_selected_audio(self, category, choice, duration_needed, sample_rate, volume, style_filter, is_opening=False):
        """加载选择的音效"""
        try:
            if choice == "无":
                return torch.zeros(1, int(duration_needed * sample_rate))

            # 获取音效文件路径
            if choice == "自动选择":
                if style_filter != "全部":
                    # 按标签筛选
                    audio_path = self.audio_effects_manager.get_ambient_sound([style_filter]) if category == "ambient" else self.audio_effects_manager.get_audio_file(category)
                else:
                    audio_path = self.audio_effects_manager.get_audio_file(category)
            else:
                # 按名称选择特定音效
                audio_path = self.audio_effects_manager.get_audio_file(category, choice)

            if not audio_path:
                print(f"🔇 {category} - {choice}: 无可用文件，使用静音")
                return torch.zeros(1, int(duration_needed * sample_rate))

            # 加载音频文件
            waveform, file_sample_rate = torchaudio.load(audio_path)

            # 重采样
            if file_sample_rate != sample_rate:
                resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
                waveform = resampler(waveform)

            # 转换为单声道
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 应用音量
            waveform = waveform * volume

            # 处理时长
            if is_opening:
                # 开场音效：只在开头播放
                track = torch.zeros(1, int(duration_needed * sample_rate))
                opening_length = min(waveform.shape[1], track.shape[1])
                track[:, :opening_length] = waveform[:, :opening_length]
                duration_display = waveform.shape[1] / sample_rate
            else:
                # 背景音乐/环境音效：循环或裁剪到指定长度
                length_needed = int(duration_needed * sample_rate)
                current_length = waveform.shape[1]

                if length_needed > current_length and category == "background":
                    # 背景音乐需要循环
                    repeats = (length_needed // current_length) + 1
                    waveform = waveform.repeat(1, repeats)

                track = waveform[:, :length_needed]
                duration_display = duration_needed

            print(f"🎵 {category} - {choice}: {duration_display:.2f}秒，音量{volume*100:.0f}%")
            return track

        except Exception as e:
            print(f"❌ 加载音效失败 ({category}-{choice}): {e}")
            return torch.zeros(1, int(duration_needed * sample_rate))

    def _combine_audio_simple(self, audio_list, voice_volume=1.0):
        """简单音频合成（回退方案）"""
        waveforms = []
        sample_rate = None

        for audio_dict in audio_list:
            waveform = audio_dict["waveform"]
            if len(waveform.shape) == 3:
                waveform = waveform[0]
            waveforms.append(waveform * voice_volume)
            sample_rate = audio_dict["sample_rate"]

        combined_waveform = torch.cat(waveforms, dim=1)

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(temp_file.name, combined_waveform, sample_rate)
        temp_file.close()

        return temp_file.name

    def _generate_effects_info(self, enabled, opening, background, ambient, style_filter):
        """生成音效配置信息"""
        if not enabled:
            return "单轨音频"

        parts = []
        if opening != "无":
            parts.append(f"开场:{opening}")
        if background != "无":
            parts.append(f"背景:{background}")
        if ambient != "无":
            parts.append(f"环境:{ambient}")

        effect_info = " | ".join(parts) if parts else "无音效"

        if style_filter != "全部":
            effect_info += f" (风格:{style_filter})"

        return effect_info

    def _get_timestamp(self):
        """获取时间戳"""
        import time
        return str(int(time.time()))

# 节点映射
NODE_CLASS_MAPPINGS = {
    "EnhancedVideoComposerV2": EnhancedVideoComposerV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedVideoComposerV2": "🖼️ 视频合成器-基于图片"
}
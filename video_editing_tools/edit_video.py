import os
import cv2
import glob
from pathlib import Path
import ffmpeg
import subprocess
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class VideoCropNode:
    """
    视频裁切节点
    输入文件夹路径，遍历所有视频文件，按指定坐标裁切并保存到目标文件夹
    """
    
    @classmethod
    def get_input_folders(cls):
        """获取输入目录下的所有子文件夹"""
        try:
            input_dir = folder_paths.get_input_directory()
            if not os.path.exists(input_dir):
                return ["input"]
            
            folders = ["input"]  # 默认包含根目录
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)
            
            return sorted(folders)
        except Exception:
            return ["input"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": (cls.get_input_folders(), {"default": "input"}),
                "output_folder_name": ("STRING", {"default": "cropped_videos", "multiline": False}),
                "crop_x1": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "crop_y1": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "crop_x2": ("INT", {"default": 1920, "min": 0, "max": 4096}),
                "crop_y2": ("INT", {"default": 1080, "min": 0, "max": 4096}),
            },
            "optional": {
                "keep_audio": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "crop_videos"
    CATEGORY = "video_editing"
    
    def crop_videos(self, input_folder, output_folder_name, crop_x1, crop_y1, crop_x2, crop_y2, keep_audio=True):
        """
        裁切视频文件
        
        Args:
            input_folder: 选择的输入子文件夹
            output_folder_name: 输出文件夹名字
            crop_x1, crop_y1: 左上角坐标
            crop_x2, crop_y2: 右下角坐标
            keep_audio: 是否保留音效
        """
        try:
            # 使用ComfyUI的默认输入和输出路径
            base_input_dir = folder_paths.get_input_directory()
            output_folder = folder_paths.get_output_directory()
            
            # 构建完整的输入路径
            if input_folder == "input":
                # 选择根目录
                input_folder_path = base_input_dir
            else:
                # 选择子文件夹
                input_folder_path = os.path.join(base_input_dir, input_folder)
            
            # 验证输入文件夹
            if not os.path.exists(input_folder_path):
                raise ValueError(f"选择的输入文件夹不存在: {input_folder_path}")
            
            # 在输出目录下创建指定名字的文件夹
            output_path = os.path.join(output_folder, output_folder_name)
            os.makedirs(output_path, exist_ok=True)
            
            # 支持的视频格式
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            
            # 遍历所有视频文件
            processed_count = 0
            output_paths = []
            for ext in video_extensions:
                pattern = os.path.join(input_folder_path, ext)
                video_files = glob.glob(pattern)
                
                for video_file in video_files:
                    try:
                        # 获取文件名（不含扩展名）
                        filename = Path(video_file).stem
                        output_file = os.path.join(output_path, f"{filename}_cropped.mp4")
                        
                        # 计算裁切宽度和高度
                        crop_width = crop_x2 - crop_x1
                        crop_height = crop_y2 - crop_y1
                        
                        # 检查原视频是否有音效
                        has_audio = False
                        try:
                            probe = ffmpeg.probe(video_file)
                            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                            has_audio = len(audio_streams) > 0
                        except Exception:
                            has_audio = False
                        
                        # 使用ffmpeg进行裁切
                        if keep_audio and has_audio:
                            # 保留音效的裁切 - 使用更明确的音视频流处理
                            input_stream = ffmpeg.input(video_file)
                            video_stream = input_stream.video.filter('crop', crop_width, crop_height, crop_x1, crop_y1)
                            audio_stream = input_stream.audio
                            
                            (
                                ffmpeg
                                .output(video_stream, audio_stream, output_file, 
                                       vcodec='libx264', acodec='aac', 
                                       audio_bitrate='128k', preset='medium')
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        else:
                            # 不保留音效的裁切
                            (
                                ffmpeg
                                .input(video_file)
                                .video
                                .filter('crop', crop_width, crop_height, crop_x1, crop_y1)
                                .output(output_file, vcodec='libx264', an=None)
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        
                        processed_count += 1
                        output_paths.append(output_file)
                        audio_status = "保留音效" if (keep_audio and has_audio) else "无音效"
                        print(f"已处理: {video_file} -> {output_file} ({audio_status})")
                        
                    except Exception as e:
                        print(f"处理视频文件 {video_file} 时出错: {str(e)}")
                        continue
            
            if processed_count == 0:
                return ("",)  # 没有可处理的视频时返回空字符串
            else:
                # 返回输出文件的目录路径
                print(f"成功处理 {processed_count} 个视频文件")
                print(f"输出目录: {output_path}")
                print(f"输出文件: {output_paths}")
                return (output_path,)
                
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            return ("",)  # 出错时也返回空字符串

class EnhancedVideoCropNode:
    """
    增强版视频裁切节点
    支持快速比例选择、自动居中计算和预览图生成
    """

    @staticmethod
    def get_input_path(input_folder):
        """获取输入文件夹的完整路径"""
        input_dir = folder_paths.get_input_directory()
        if input_folder == "input":
            return input_dir
        else:
            return os.path.join(input_dir, input_folder)

    @classmethod
    def get_input_folders(cls):
        """获取输入目录下的所有子文件夹"""
        try:
            input_dir = folder_paths.get_input_directory()
            if not os.path.exists(input_dir):
                return ["input"]

            folders = ["input"]  # 默认包含根目录
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)

            return sorted(folders)
        except Exception:
            return ["input"]


    @classmethod
    def get_aspect_ratios(cls):
        """获取预设的宽高比选项"""
        return {
            "自定义": None,
            "16:9": (16, 9),
            "9:16": (9, 16),
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "21:9": (21, 9),
            "2:1": (2, 1),
            "3:2": (3, 2),
            "2:3": (2, 3)
        }

    @classmethod
    def calculate_crop_coordinates(cls, video_width, video_height, aspect_ratio_key, offset_x=0, offset_y=0):
        """
        根据视频尺寸和目标宽高比计算裁切坐标

        Args:
            video_width: 视频宽度
            video_height: 视频高度
            aspect_ratio_key: 宽高比选择
            offset_x: X轴偏移
            offset_y: Y轴偏移

        Returns:
            tuple: (crop_x1, crop_y1, crop_x2, crop_y2, crop_width, crop_height)
        """
        aspect_ratios = cls.get_aspect_ratios()

        if aspect_ratio_key == "自定义" or aspect_ratio_key not in aspect_ratios:
            # 自定义模式，返回原始尺寸
            return 0, 0, video_width, video_height, video_width, video_height

        target_ratio = aspect_ratios[aspect_ratio_key]
        target_width_ratio, target_height_ratio = target_ratio

        # 计算目标宽高比
        target_aspect = target_width_ratio / target_height_ratio
        current_aspect = video_width / video_height

        if current_aspect > target_aspect:
            # 视频更宽，以高度为基准
            crop_height = video_height
            crop_width = int(crop_height * target_aspect)
        else:
            # 视频更高，以宽度为基准
            crop_width = video_width
            crop_height = int(crop_width / target_aspect)

        # 确保裁切尺寸不超过原视频尺寸
        crop_width = min(crop_width, video_width)
        crop_height = min(crop_height, video_height)

        # 修复libx264要求：确保宽高都是偶数
        crop_width = crop_width if crop_width % 2 == 0 else crop_width - 1
        crop_height = crop_height if crop_height % 2 == 0 else crop_height - 1

        # 计算居中坐标
        crop_x1 = (video_width - crop_width) // 2 + offset_x
        crop_y1 = (video_height - crop_height) // 2 + offset_y

        # 确保坐标在有效范围内
        crop_x1 = max(0, min(crop_x1, video_width - crop_width))
        crop_y1 = max(0, min(crop_y1, video_height - crop_height))

        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        return crop_x1, crop_y1, crop_x2, crop_y2, crop_width, crop_height


    @classmethod
    def detect_video_resolution(cls, input_folder):
        """
        自动探测输入文件夹中第一个视频的分辨率
        """
        try:
            input_path = EnhancedVideoCropNode.get_input_path(input_folder)
            if not os.path.exists(input_path):
                return 1920, 1080  # 默认分辨率

            # 查找第一个视频文件
            for filename in sorted(os.listdir(input_path)):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                    video_file = os.path.join(input_path, filename)
                    try:
                        # 使用ffprobe获取视频信息
                        probe = ffmpeg.probe(video_file)
                        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                        if video_stream:
                            width = int(video_stream['width'])
                            height = int(video_stream['height'])
                            print(f"📐 检测到视频分辨率: {width}×{height} (文件: {filename})")
                            return width, height
                    except Exception as e:
                        print(f"⚠️ 无法读取视频 {filename}: {e}")
                        continue

            print("⚠️ 未找到有效视频文件，使用默认分辨率 1920×1080")
            return 1920, 1080
        except Exception as e:
            print(f"⚠️ 探测视频分辨率失败: {e}，使用默认分辨率")
            return 1920, 1080

    @classmethod
    def extract_video_frame(cls, input_folder, frame_time=1.0):
        """
        提取视频首帧用于预览
        Args:
            input_folder: 输入文件夹
            frame_time: 提取帧的时间点（秒）
        Returns:
            (frame_path, video_width, video_height) 或 (None, None, None)
        """
        try:
            input_path = EnhancedVideoCropNode.get_input_path(input_folder)
            if not os.path.exists(input_path):
                return None, None, None

            # 查找第一个视频文件
            for filename in sorted(os.listdir(input_path)):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                    video_file = os.path.join(input_path, filename)
                    try:
                        # 使用ComfyUI的输出目录来存储预览帧（可被Web访问）
                        output_dir = folder_paths.get_output_directory()
                        cache_dir = os.path.join(output_dir, "video_previews")
                        os.makedirs(cache_dir, exist_ok=True)

                        # 生成唯一的帧文件名，包含路径哈希避免同名文件冲突
                        import hashlib
                        path_hash = hashlib.md5(video_file.encode('utf-8')).hexdigest()[:8]
                        frame_filename = f"{filename}_{path_hash}_{int(frame_time)}s_frame.jpg"
                        frame_path = os.path.join(cache_dir, frame_filename)

                        # 检查帧文件是否已存在且比视频文件新
                        frame_cached = False
                        if os.path.exists(frame_path):
                            video_mtime = os.path.getmtime(video_file)
                            frame_mtime = os.path.getmtime(frame_path)
                            if frame_mtime > video_mtime:
                                # 缓存有效，跳过帧生成
                                frame_cached = True
                                print(f"🎯 使用缓存的预览帧: {frame_path}")
                            else:
                                print(f"🔄 视频文件已更新，重新生成预览帧: {video_file}")

                        # 如果没有缓存，提取视频帧
                        if not frame_cached:
                            (
                                ffmpeg
                                .input(video_file, ss=frame_time)
                                .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                                .overwrite_output()
                                .run(quiet=True)
                            )

                        # 获取视频分辨率
                        probe = ffmpeg.probe(video_file)
                        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                        if video_stream:
                            width = int(video_stream['width'])
                            height = int(video_stream['height'])
                            if not frame_cached:
                                print(f"📸 提取视频帧: {frame_path} ({width}×{height})")
                            else:
                                print(f"📸 使用缓存帧: {frame_path} ({width}×{height})")

                            # 额外生成JavaScript期望的预览图片文件，包含路径哈希避免同名目录冲突
                            input_folder_full_path = EnhancedVideoCropNode.get_input_path(input_folder)
                            input_path_hash = hashlib.sha256(input_folder_full_path.encode('utf-8')).hexdigest()[:8]
                            preview_filename = f"video_preview_{input_folder}_{input_path_hash}.jpg"
                            preview_path_subdir = os.path.join(cache_dir, preview_filename)
                            preview_path_root = os.path.join(output_dir, preview_filename)

                            try:
                                # 检查预览图片是否需要更新
                                import shutil
                                need_update = True

                                if os.path.exists(preview_path_root):
                                    frame_mtime = os.path.getmtime(frame_path)
                                    preview_mtime = os.path.getmtime(preview_path_root)
                                    need_update = frame_mtime > preview_mtime

                                if need_update:
                                    # 复制帧文件到预览位置（在video_previews子目录）
                                    shutil.copy2(frame_path, preview_path_subdir)
                                    print(f"📸 预览图片已生成: {preview_path_subdir}")

                                    # 同时在output根目录也生成一份供JavaScript访问
                                    shutil.copy2(frame_path, preview_path_root)
                                    print(f"📸 JavaScript预览图片已生成: {preview_path_root}")
                                else:
                                    print(f"🎯 使用缓存的预览图片: {preview_path_root}")
                            except Exception as e:
                                print(f"⚠️ 生成预览图片失败: {e}")

                            return frame_path, width, height

                    except Exception as e:
                        print(f"⚠️ 无法提取视频帧 {filename}: {e}")
                        continue

            return None, None, None
        except Exception as e:
            print(f"⚠️ 提取视频帧失败: {e}")
            return None, None, None



    @classmethod
    def generate_preview_video(cls, video_path, crop_coords, output_path, duration_limit=10):
        """
        生成带有裁切框和遮罩的预览视频

        Args:
            video_path: 原视频文件路径
            crop_coords: 裁切坐标 (x1, y1, x2, y2)
            output_path: 预览视频保存路径
            duration_limit: 预览视频时长限制（秒）
        """
        try:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1

            # 使用ffmpeg创建预览视频
            input_stream = ffmpeg.input(video_path, t=duration_limit)  # 限制预览时长

            # 创建带遮罩和边框的视频
            # 1. 创建半透明遮罩层
            mask_filter = (
                input_stream
                .video
                .filter('drawbox',
                       x=0, y=0,
                       w='iw', h='ih',
                       color='black@0.5',
                       thickness='fill')
                .filter('drawbox',
                       x=crop_x1, y=crop_y1,
                       w=crop_width, h=crop_height,
                       color='black@0.0',
                       thickness='fill')
            )

            # 2. 添加红色边框
            preview_with_border = (
                mask_filter
                .filter('drawbox',
                       x=crop_x1, y=crop_y1,
                       w=crop_width, h=crop_height,
                       color='red',
                       thickness=3)
            )

            # 获取原视频尺寸信息
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            orig_width = int(video_stream['width'])
            orig_height = int(video_stream['height'])

            # 3. 添加信息文本
            text_filter = (
                preview_with_border
                .filter('drawtext',
                       text=f'原尺寸: {orig_width}x{orig_height}\\n裁切: {crop_width}x{crop_height}\\n坐标: ({crop_x1},{crop_y1})',
                       x=10, y=10,
                       fontsize=20,
                       fontcolor='white',
                       box=1,
                       boxcolor='black@0.8',
                       boxborderw=5)
            )

            # 输出预览视频
            (
                ffmpeg
                .output(text_filter, output_path,
                       vcodec='libx264',
                       preset='fast',
                       crf=23,
                       an=None)  # 不包含音频
                .overwrite_output()
                .run(quiet=True)
            )

            return True

        except Exception as e:
            print(f"生成预览视频失败: {str(e)}")
            return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": (EnhancedVideoCropNode.get_input_folders(), {"default": "input"}),
                "output_folder_name": ("STRING", {"default": "cropped_videos", "multiline": False}),
                "aspect_ratio": (list(cls.get_aspect_ratios().keys()), {"default": "16:9"}),
            },
            "optional": {
                # 裁切位置和尺寸参数
                "pos_x": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "裁切区域左上角X坐标"}),
                "pos_y": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "裁切区域左上角Y坐标"}),
                "crop_width": ("INT", {"default": 1920, "min": 1, "max": 4096, "tooltip": "裁切区域宽度"}),
                "crop_height": ("INT", {"default": 1080, "min": 1, "max": 4096, "tooltip": "裁切区域高度"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "enhanced_crop_videos"
    CATEGORY = "video_editing"

    def enhanced_crop_videos(self, input_folder, output_folder_name, aspect_ratio,
                           pos_x=0, pos_y=0, crop_width=1920, crop_height=1080):
        """
        增强版视频裁切功能
        默认启用预览模式和保留音频
        """
        try:
            # 直接设置为生产模式，不只是预览
            preview_only = False  # 直接生产模式，不只是预览
            keep_audio = True     # 默认保留音频
            # 自动探测视频分辨率
            video_width, video_height = self.detect_video_resolution(input_folder)
            print(f"🔍 自动探测视频分辨率: {video_width}×{video_height}")

            # 自动生成预览帧用于前端显示
            frame_path, frame_width, frame_height = self.extract_video_frame(input_folder)
            if frame_path:
                print(f"📸 视频预览帧已生成: {frame_path}")
            else:
                print("⚠️ 未能提取视频帧")
            # 获取输入输出路径
            input_folder_path = EnhancedVideoCropNode.get_input_path(input_folder)
            if not os.path.exists(input_folder_path):
                raise ValueError(f"输入文件夹不存在: {input_folder_path}")

            output_folder = folder_paths.get_output_directory()
            output_path = os.path.join(output_folder, output_folder_name)
            os.makedirs(output_path, exist_ok=True)

            # 创建预览文件夹（始终生成预览视频）
            preview_path = os.path.join(output_path, "previews")
            os.makedirs(preview_path, exist_ok=True)

            # 支持的视频格式
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']

            processed_count = 0
            preview_count = 0

            for ext in video_extensions:
                pattern = os.path.join(input_folder_path, ext)
                video_files = glob.glob(pattern)

                for video_file in video_files:
                    try:
                        filename = Path(video_file).stem

                        # 获取视频信息
                        probe = ffmpeg.probe(video_file)
                        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

                        if not video_stream:
                            print(f"无法获取视频流信息: {video_file}")
                            continue

                        actual_video_width = int(video_stream['width'])
                        actual_video_height = int(video_stream['height'])

                        # 验证实际视频分辨率
                        if actual_video_width != video_width or actual_video_height != video_height:
                            print(f"⚠️ 视频 {filename} 分辨率 {actual_video_width}×{actual_video_height} 与探测分辨率 {video_width}×{video_height} 不匹配，使用实际分辨率")

                        # 使用自定义坐标模式
                        final_x1, final_y1 = pos_x, pos_y
                        final_crop_width, final_crop_height = crop_width, crop_height
                        final_x2 = final_x1 + final_crop_width
                        final_y2 = final_y1 + final_crop_height

                        # 验证坐标有效性
                        if (final_x1 >= final_x2 or final_y1 >= final_y2 or
                            final_x2 > actual_video_width or final_y2 > actual_video_height or
                            final_x1 < 0 or final_y1 < 0):
                            print(f"无效的裁切坐标: {video_file}, 坐标: ({final_x1},{final_y1}) → ({final_x2},{final_y2}), 视频尺寸: {actual_video_width}×{actual_video_height}")
                            continue

                        # 生成10秒预览视频
                        preview_file = os.path.join(preview_path, f"{filename}_preview.mp4")
                        if self.generate_preview_video(video_file, (final_x1, final_y1, final_x2, final_y2), preview_file, 10):
                            preview_count += 1
                            print(f"预览视频已生成: {preview_file} (时长: 10秒)")

                        # 如果只是预览模式，跳过视频处理
                        if preview_only:
                            continue

                        # 处理视频裁切
                        output_file = os.path.join(output_path, f"{filename}_cropped.mp4")

                        # 检查音频流
                        has_audio = False
                        try:
                            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                            has_audio = len(audio_streams) > 0
                        except Exception:
                            has_audio = False

                        # 执行裁切
                        if keep_audio and has_audio:
                            input_stream = ffmpeg.input(video_file)
                            video_stream = input_stream.video.filter('crop', final_crop_width, final_crop_height, final_x1, final_y1)
                            audio_stream = input_stream.audio

                            (
                                ffmpeg
                                .output(video_stream, audio_stream, output_file,
                                       vcodec='libx264', acodec='aac',
                                       audio_bitrate='128k', preset='medium')
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        else:
                            (
                                ffmpeg
                                .input(video_file)
                                .video
                                .filter('crop', final_crop_width, final_crop_height, final_x1, final_y1)
                                .output(output_file, vcodec='libx264', an=None)
                                .overwrite_output()
                                .run(quiet=True)
                            )

                        processed_count += 1
                        audio_status = "保留音效" if (keep_audio and has_audio) else "无音效"
                        crop_info = f"裁切尺寸: {final_crop_width}×{final_crop_height}"
                        print(f"已处理: {filename} -> {crop_info} ({audio_status})")

                    except Exception as e:
                        print(f"处理视频文件 {video_file} 时出错: {str(e)}")
                        continue

            # 生成结果报告
            result_parts = []

            if preview_count > 0:
                result_parts.append(f"生成预览视频: {preview_count} 个")
            if processed_count > 0:
                result_parts.append(f"裁切视频: {processed_count} 个")

            # 打印处理结果
            if result_parts:
                print(f"✅ 处理完成: {', '.join(result_parts)}")

            if preview_only:
                if preview_count == 0:
                    return (output_path,)  # 即使没有找到文件也返回输出路径
                return (os.path.join(output_path, 'previews'),)  # 返回预览文件夹路径

            # 实际裁切模式：返回主输出文件夹路径
            return (output_path,)

        except Exception as e:
            return (f"处理过程中出错: {str(e)}",)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "EnhancedVideoCropNode": EnhancedVideoCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedVideoCropNode": "批量视频画面裁切"
}

# Web目录配置
WEB_DIRECTORY = "./web"
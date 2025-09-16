"""
视频预览节点
支持在ComfyUI界面中直接预览视频文件
"""

import os
import tempfile
import shutil
import folder_paths
import json
import base64
from typing import Optional

class VideoPreview:
    """
    视频预览节点

    接收视频文件路径，在ComfyUI界面中显示视频预览
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "视频文件路径"
                }),
            },
            "optional": {
                "autoplay": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自动播放"
                }),
                "loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "循环播放"
                }),
                "controls": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "显示播放控制器"
                })
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "🎬 Shenglin/Video"
    DESCRIPTION = "在ComfyUI界面中预览视频"

    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    def preview_video(self, video_path: str, autoplay: bool = True,
                     loop: bool = False, controls: bool = True,
                     prompt=None, extra_pnginfo=None):
        """
        预览视频的主函数
        """
        try:
            if not video_path or not os.path.exists(video_path):
                return {
                    "ui": {
                        "video": [{
                            "error": "视频文件不存在或路径无效",
                            "path": video_path
                        }]
                    }
                }

            # 获取视频文件信息
            file_size = os.path.getsize(video_path)
            file_name = os.path.basename(video_path)

            print(f"🎥 预览视频: {file_name} ({file_size / (1024*1024):.2f} MB)")

            # 检查视频文件是否可访问
            if not self._is_video_file(video_path):
                return {
                    "ui": {
                        "video": [{
                            "error": "不支持的视频格式",
                            "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"]
                        }]
                    }
                }

            # 生成预览结果
            video_info = self._prepare_video_preview(
                video_path, file_name, file_size, autoplay, loop, controls
            )

            return {"ui": {"video": [video_info]}}

        except Exception as e:
            error_msg = f"视频预览失败: {str(e)}"
            print(f"❌ {error_msg}")

            return {
                "ui": {
                    "video": [{
                        "error": error_msg
                    }]
                }
            }

    def _is_video_file(self, file_path: str) -> bool:
        """检查是否为支持的视频文件"""
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in supported_extensions

    def _prepare_video_preview(self, video_path: str, file_name: str,
                              file_size: int, autoplay: bool,
                              loop: bool, controls: bool) -> dict:
        """
        准备视频预览数据
        """
        # 检查是否需要复制到临时目录（如果视频不在可访问路径）
        accessible_path = self._ensure_accessible_path(video_path, file_name)

        # 生成相对于ComfyUI的访问路径
        output_dir = folder_paths.get_output_directory()

        if accessible_path.startswith(output_dir):
            # 输出目录中的文件
            relative_path = os.path.relpath(accessible_path, output_dir)
            web_path = f"/view?filename={relative_path}&type=output&subfolder="
        elif accessible_path.startswith(self.temp_dir):
            # 临时目录中的文件
            relative_path = os.path.relpath(accessible_path, self.temp_dir)
            web_path = f"/view?filename={relative_path}&type=temp&subfolder="
        else:
            # 复制到temp目录
            temp_path = os.path.join(self.temp_dir, file_name)
            shutil.copy2(accessible_path, temp_path)
            web_path = f"/view?filename={file_name}&type=temp&subfolder="

        # 获取视频元数据（如果可能）
        video_info = self._get_video_metadata(accessible_path)

        return {
            "filename": file_name,
            "path": web_path,
            "type": "video",
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "autoplay": autoplay,
            "loop": loop,
            "controls": controls,
            "format": "video/mp4",  # 默认格式，前端会自动检测
            **video_info
        }

    def _ensure_accessible_path(self, video_path: str, file_name: str) -> str:
        """确保视频文件在可访问的路径"""
        temp_dir = folder_paths.get_temp_directory()
        output_dir = folder_paths.get_output_directory()

        # 如果已经在temp或output目录，直接使用
        if video_path.startswith(temp_dir) or video_path.startswith(output_dir):
            return video_path

        # 否则复制到temp目录
        temp_path = os.path.join(temp_dir, file_name)
        if not os.path.exists(temp_path) or os.path.getmtime(video_path) > os.path.getmtime(temp_path):
            print(f"📁 复制视频到临时目录: {temp_path}")
            shutil.copy2(video_path, temp_path)

        return temp_path

    def _get_video_metadata(self, video_path: str) -> dict:
        """
        获取视频元数据
        """
        metadata = {}

        try:
            # 尝试使用ffprobe获取视频信息
            import subprocess
            import json as json_lib

            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                probe_data = json_lib.loads(result.stdout)

                # 获取视频流信息
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        metadata.update({
                            'width': stream.get('width'),
                            'height': stream.get('height'),
                            'duration': float(stream.get('duration', 0)),
                            'fps': eval(stream.get('r_frame_rate', '30/1'))
                        })
                        break

                # 获取格式信息
                format_info = probe_data.get('format', {})
                if 'duration' in format_info:
                    metadata['duration'] = float(format_info['duration'])

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, Exception):
            # ffprobe不可用或执行失败，使用默认值
            pass

        return metadata


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoPreview": VideoPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPreview": "🎥 视频预览"
}
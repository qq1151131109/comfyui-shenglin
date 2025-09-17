"""
视频预览节点
支持在ComfyUI界面中直接预览视频文件
"""

import os
import folder_paths
import shutil
from pathlib import Path

# 尝试导入新的ComfyUI API
try:
    from comfy_api.latest import ui, io
    NEW_API_AVAILABLE = True
    print("✅ VideoPreview: 使用新的ComfyUI API")
except ImportError:
    NEW_API_AVAILABLE = False
    print("⚠️ VideoPreview: 使用传统API格式")

class VideoPreview:
    """
    视频预览节点

    接收视频文件路径，在ComfyUI界面中显示视频预览
    使用ComfyUI标准的视频预览机制
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
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "🔥 Shenglin/视频处理"
    DESCRIPTION = "在ComfyUI界面中预览视频"

    def preview_video(self, video_path: str, prompt=None, extra_pnginfo=None):
        """
        预览视频的主函数
        """
        try:
            if not video_path or not os.path.exists(video_path):
                print(f"❌ 视频文件不存在: {video_path}")
                return {"ui": {"text": [f"视频文件不存在: {video_path}"]}}

            # 获取视频文件信息
            file_name = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)

            print(f"🎥 预览视频: {file_name} ({file_size / (1024*1024):.2f} MB)")

            # 检查视频文件格式
            if not self._is_video_file(video_path):
                supported_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
                print(f"❌ 不支持的视频格式，支持的格式: {supported_formats}")
                return {"ui": {"text": [f"不支持的视频格式，支持的格式: {supported_formats}"]}}

            # 确保视频文件在可访问的路径
            accessible_path, subfolder, folder_type = self._ensure_accessible_path(video_path, file_name)

            # 使用ComfyUI标准的视频预览格式 (视频作为动画图像处理)
            if NEW_API_AVAILABLE:
                # 使用新的API格式：PreviewVideo返回 {"images": [...], "animated": (True,)}
                folder_type_enum = io.FolderType.output if folder_type == "output" else io.FolderType.temp
                saved_result = ui.SavedResult(
                    filename=os.path.basename(accessible_path),
                    subfolder=subfolder,
                    type=folder_type_enum
                )
                preview_video = ui.PreviewVideo([saved_result])
                result_ui = preview_video.as_dict()
                print(f"✅ 视频预览准备完成 (新API): {result_ui}")
                return {"ui": result_ui}
            else:
                # 使用传统格式：视频作为animated images
                result = {
                    "filename": os.path.basename(accessible_path),
                    "subfolder": subfolder,
                    "type": folder_type
                }
                print(f"✅ 视频预览准备完成 (传统API): {result}")
                # ComfyUI视频预览的正确格式：images + animated标志
                return {
                    "ui": {
                        "images": [result],
                        "animated": (True,)
                    }
                }

        except Exception as e:
            error_msg = f"视频预览失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {"ui": {"text": [error_msg]}}

    def _is_video_file(self, file_path: str) -> bool:
        """检查是否为支持的视频文件"""
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv']
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in supported_extensions

    def _ensure_accessible_path(self, video_path: str, file_name: str) -> tuple:
        """
        确保视频文件在ComfyUI可访问的路径
        返回: (accessible_path, subfolder, folder_type)
        """
        temp_dir = folder_paths.get_temp_directory()
        output_dir = folder_paths.get_output_directory()

        # 检查文件是否已经在output目录
        if video_path.startswith(output_dir):
            relative_path = os.path.relpath(video_path, output_dir)
            subfolder = os.path.dirname(relative_path) if os.path.dirname(relative_path) else ""
            return video_path, subfolder, "output"

        # 检查文件是否已经在temp目录
        if video_path.startswith(temp_dir):
            relative_path = os.path.relpath(video_path, temp_dir)
            subfolder = os.path.dirname(relative_path) if os.path.dirname(relative_path) else ""
            return video_path, subfolder, "temp"

        # 文件不在可访问目录，复制到temp目录
        # 保持原始文件名，避免重复
        temp_path = os.path.join(temp_dir, file_name)

        # 如果文件已存在且是同一个文件，不需要重复复制
        if os.path.exists(temp_path):
            if os.path.getmtime(video_path) <= os.path.getmtime(temp_path):
                print(f"📁 使用已存在的临时文件: {temp_path}")
                return temp_path, "", "temp"

        # 复制文件到temp目录
        print(f"📁 复制视频到临时目录: {temp_path}")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        shutil.copy2(video_path, temp_path)

        return temp_path, "", "temp"


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoPreview": VideoPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPreview": "🎥 视频预览"
}
"""
批量文件上传节点
支持批量上传视频/音频/图片到ComfyUI input目录
自动创建带时间戳的子目录
"""
import os
import shutil
import glob
from datetime import datetime
import folder_paths


class BatchFileUploader:
    """批量文件上传器"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("COMBO", {
                    "image_upload": True,
                    "tooltip": "点击📁按钮上传文件"
                }),
                "folder_prefix": ("STRING", {
                    "default": "uploaded",
                    "tooltip": "文件夹前缀，会自动加时间戳\n例如: videos → videos_20241001_123456"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder_path", "status")
    FUNCTION = "organize_uploaded_files"
    CATEGORY = "🔥 Shenglin/素材上传与下载"

    @classmethod
    def IS_CHANGED(cls, image, folder_prefix):
        # 每次都执行，因为可能有新上传的文件
        import time
        return time.time()

    def organize_uploaded_files(self, image, folder_prefix="uploaded"):
        """整理上传的文件到子文件夹"""

        input_directory = folder_paths.get_input_directory()

        # 1. 创建目标文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 清理文件夹前缀中的非法字符
        folder_prefix = "".join(c for c in folder_prefix if c.isalnum() or c in "._-")
        folder_name = f"{folder_prefix}_{timestamp}"
        target_folder = os.path.join(input_directory, folder_name)

        os.makedirs(target_folder, exist_ok=True)

        # 2. 查找最近上传的文件（通过修改时间判断，最近10秒内的）
        import time
        current_time = time.time()

        all_files = []
        for f in os.listdir(input_directory):
            file_path = os.path.join(input_directory, f)
            if os.path.isfile(file_path):
                mtime = os.path.getmtime(file_path)
                # 只获取最近10秒内修改的文件（刚上传的）
                if current_time - mtime < 10:
                    all_files.append((f, mtime, file_path))

        if not all_files:
            # 如果没有找到最近上传的文件
            status_msg = "⚠️ 未检测到新上传的文件\n\n"
            status_msg += "💡 提示：请先点击📁按钮上传文件，然后立即执行节点"
            return (target_folder, status_msg)

        # 按修改时间排序
        all_files.sort(key=lambda x: x[1], reverse=True)

        # 3. 移动文件到目标文件夹
        moved_files = []
        failed_files = []

        for filename, mtime, file_path in all_files:
            try:
                target_path = os.path.join(target_folder, filename)

                # 移动文件
                shutil.move(file_path, target_path)
                moved_files.append(filename)
                print(f"✓ 已移动: {filename} -> {folder_name}/")

            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                failed_files.append(error_msg)
                print(f"✗ 移动失败: {error_msg}")

        # 4. 生成状态报告
        status_msg = self._generate_summary(
            folder_name, target_folder, moved_files, failed_files
        )

        return (target_folder, status_msg)

    def _generate_summary(self, folder_name, folder_path, uploaded_files, failed_files):
        """生成上传报告"""
        msg = f"{'='*50}\n"
        msg += f"📤 批量上传完成!\n"
        msg += f"{'='*50}\n\n"
        msg += f"📊 统计信息:\n"
        msg += f"  文件夹: {folder_name}\n"
        msg += f"  上传成功: {len(uploaded_files)}个\n"
        msg += f"  上传失败: {len(failed_files)}个\n"

        total = len(uploaded_files) + len(failed_files)
        if total > 0:
            msg += f"  成功率: {len(uploaded_files)/total*100:.1f}%\n"

        msg += f"\n📁 目标目录: {folder_path}\n"

        if uploaded_files:
            msg += f"\n✅ 已上传文件（显示前20个）:\n"
            for f in uploaded_files[:20]:
                msg += f"  - {f}\n"
            if len(uploaded_files) > 20:
                msg += f"  ... 还有 {len(uploaded_files) - 20} 个文件\n"

        if failed_files:
            msg += f"\n❌ 失败文件:\n"
            for f in failed_files[:10]:
                msg += f"  - {f}\n"
            if len(failed_files) > 10:
                msg += f"  ... 还有 {len(failed_files) - 10} 个错误\n"

        return msg


NODE_CLASS_MAPPINGS = {
    "批量文件上传器": BatchFileUploader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "批量文件上传器": "📤 批量文件上传器",
}

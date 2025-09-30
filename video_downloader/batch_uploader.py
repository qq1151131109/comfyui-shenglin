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
        # 获取input目录下的所有文件
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "files": (sorted(files), {
                    "image_upload": True,
                    "tooltip": "点击上传文件（支持多选）"
                }),
                "folder_prefix": ("STRING", {
                    "default": "uploaded",
                    "tooltip": "文件夹名称前缀，后面会自动加上时间戳"
                }),
                "organize_files": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否将上传的文件整理到带时间戳的子文件夹"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder_path", "status")
    FUNCTION = "organize_uploaded_files"
    CATEGORY = "🔥 Shenglin/视频下载"

    def organize_uploaded_files(self, files, folder_prefix="uploaded", organize_files=True):
        """整理上传的文件到子文件夹"""

        input_directory = folder_paths.get_input_directory()

        # 如果不需要整理，直接返回input目录
        if not organize_files:
            file_path = folder_paths.get_annotated_filepath(files)
            status_msg = f"📁 文件已上传到: {input_directory}\n"
            status_msg += f"📄 文件名: {files}\n"
            status_msg += f"📍 完整路径: {file_path}"
            return (input_directory, status_msg)

        # 1. 创建目标文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 清理文件夹前缀中的非法字符
        folder_prefix = "".join(c for c in folder_prefix if c.isalnum() or c in "._-")
        folder_name = f"{folder_prefix}_{timestamp}"
        target_folder = os.path.join(input_directory, folder_name)

        os.makedirs(target_folder, exist_ok=True)

        # 2. 查找最近上传的文件（通过修改时间判断）
        # 获取所有文件及其修改时间
        all_files = []
        for f in os.listdir(input_directory):
            file_path = os.path.join(input_directory, f)
            if os.path.isfile(file_path):
                mtime = os.path.getmtime(file_path)
                all_files.append((f, mtime, file_path))

        # 按修改时间排序，获取最近的文件
        all_files.sort(key=lambda x: x[1], reverse=True)

        # 获取最近10秒内修改的文件（可能是刚上传的）
        import time
        current_time = time.time()
        recent_files = [f for f in all_files if current_time - f[1] < 10]

        if not recent_files:
            # 如果没有最近上传的文件，至少移动当前选中的文件
            file_path = folder_paths.get_annotated_filepath(files)
            recent_files = [(files, 0, file_path)]

        # 3. 移动文件到目标文件夹
        moved_files = []
        failed_files = []

        for filename, mtime, file_path in recent_files:
            try:
                if os.path.basename(file_path) == os.path.basename(target_folder):
                    # 跳过目标文件夹本身
                    continue

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

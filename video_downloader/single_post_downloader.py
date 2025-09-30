"""
单个帖子下载节点
提取并下载单个或多个视频/图片/音乐帖子
支持多URL批量下载，自动创建子目录
"""
import requests
import json
import os
import time
import urllib3
import folder_paths

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SinglePostDownloader:
    """哼哼猫-单个帖子下载节点"""

    API_URL = "https://h.aaaapp.cn/single_post"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "单个帖子URL"
                }),
                "user_id": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),
                "media_types": (["all", "video", "image", "audio"], {"default": "video"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_folder", "status")
    FUNCTION = "download_post"
    CATEGORY = "🔥 Shenglin/视频下载"

    def download_post(self, url, user_id, secret_key, media_types="video"):
        """提取并下载单个帖子"""
        # 固定参数：质量最高、重试6次、延迟2秒
        quality_priority = "highest"
        retry_times = 6
        retry_delay = 2
        # 1. 验证参数
        if not url or not url.strip():
            return ("", "错误: URL不能为空")
        if not user_id or not secret_key:
            return ("", "错误: 请配置userId和secretKey")

        # 2. 提取帖子信息
        try:
            params = {
                "userId": user_id,
                "secretKey": secret_key,
                "url": url
            }

            response = requests.post(
                self.API_URL,
                json=params,
                timeout=30,
                verify=False
            )

            result = response.json()

            if result.get("code") != 200 or not result.get("succ"):
                code = result.get("code", "unknown")
                msg = result.get("msg", "未知错误")
                return ("", f"提取失败 (code: {code}): {msg}")

            data = result.get("data", {})
            text = data.get("text", "")
            medias = data.get("medias", [])
            user_info = data.get("user", {})
            username = user_info.get("username", "unknown_user")
            user_id_str = user_info.get("user_id", "")

            if not medias:
                return ("", "提取成功但没有找到媒体资源")

            # 过滤媒体类型
            if media_types != "all":
                medias = [m for m in medias if m.get("media_type") == media_types]

            if not medias:
                return ("", f"没有找到{media_types}类型的媒体")

            # 3. 生成输出目录（使用 user_id 或 username 作为前缀）
            from datetime import datetime
            output_base_folder = folder_paths.get_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 使用 user_id 或 username 作为文件夹前缀
            folder_prefix = user_id_str if user_id_str else username
            # 清理文件名中的非法字符
            folder_prefix = "".join(c for c in folder_prefix if c.isalnum() or c in "._-")
            folder_name = f"{folder_prefix}_{timestamp}"
            output_folder = os.path.join(output_base_folder, folder_name)

            os.makedirs(output_folder, exist_ok=True)

            # 4. 下载媒体
            success_count = 0
            fail_count = 0
            downloaded_files = []

            for idx, media in enumerate(medias):
                media_type = media.get("media_type", "unknown")
                download_url = self._get_download_url(media, quality_priority)

                if not download_url:
                    fail_count += 1
                    continue

                # 生成文件名
                ext = self._get_extension(download_url, media_type)
                filename = f"{media_type}_{int(time.time())}_{idx}{ext}"
                filepath = os.path.join(output_folder, filename)

                # 下载
                headers = media.get("headers", {})
                if not headers:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }

                try:
                    resp = requests.get(
                        download_url,
                        headers=headers,
                        timeout=300,
                        stream=True,
                        verify=False
                    )

                    if resp.status_code == 200:
                        with open(filepath, 'wb') as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                        success_count += 1
                        downloaded_files.append(filename)
                        print(f"✓ 下载成功: {filename}")
                    else:
                        fail_count += 1
                        print(f"✗ 下载失败: {filename} (HTTP {resp.status_code})")

                except Exception as e:
                    fail_count += 1
                    print(f"✗ 下载失败: {filename} - {str(e)}")

            # 5. 生成状态报告
            status_msg = f"✓ 下载完成!\n"
            status_msg += f"作者: {username}\n"
            status_msg += f"文案: {text[:50]}...\n" if len(text) > 50 else f"文案: {text}\n"
            status_msg += f"成功: {success_count}个 | 失败: {fail_count}个\n"
            status_msg += f"输出目录: {output_folder}\n"

            if downloaded_files:
                status_msg += f"\n已下载文件:\n"
                for f in downloaded_files:
                    status_msg += f"  - {f}\n"

            return (output_folder, status_msg)

        except requests.exceptions.Timeout:
            return ("", "错误: 请求超时")
        except requests.exceptions.RequestException as e:
            return ("", f"错误: {str(e)}")
        except Exception as e:
            return ("", f"错误: {str(e)}")

    def _get_download_url(self, media, quality_priority):
        """获取下载URL（支持多清晰度选择）"""
        if "formats" in media:
            formats = media["formats"]
            if quality_priority == "highest":
                selected = max(formats, key=lambda x: x.get("quality", 0))
            elif quality_priority == "lowest":
                selected = min(formats, key=lambda x: x.get("quality", 9999))
            else:  # medium
                sorted_formats = sorted(formats, key=lambda x: x.get("quality", 0))
                selected = sorted_formats[len(sorted_formats) // 2]

            # 简化处理：只下载视频部分（完整实现需要合并音视频）
            if selected.get("separate") == 1:
                return selected.get("video_url")
            else:
                return selected.get("video_url")
        else:
            return media.get("resource_url")

    def _get_extension(self, url, media_type):
        """从URL获取文件扩展名"""
        path = url.split('?')[0]
        ext = os.path.splitext(path)[1]

        if ext:
            return ext

        defaults = {
            "video": ".mp4",
            "image": ".jpg",
            "audio": ".m4a"
        }
        return defaults.get(media_type, ".bin")


NODE_CLASS_MAPPINGS = {
    "哼哼猫-单个帖子下载": SinglePostDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "哼哼猫-单个帖子下载": "哼哼猫-单个帖子下载",
}
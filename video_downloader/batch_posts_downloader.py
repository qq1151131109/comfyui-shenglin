"""
主页批量下载节点
提取并下载主页/播放列表/频道/话题的多个帖子
支持自动翻页和重试机制
支持多URL批量下载，自动创建子目录
"""
import requests
import json
import os
import time
import urllib3
import folder_paths

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BatchPostsDownloader:
    """哼哼猫-主页批量下载节点"""

    API_URL = "https://h.aaaapp.cn/posts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "主页/播放列表/频道/话题URL"
                }),
                "user_id": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),
                "media_types": (["all", "video", "image", "audio"], {"default": "video"}),
                "max_videos": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "最大下载视频数量，默认100个"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_folder", "status")
    FUNCTION = "download_batch"
    CATEGORY = "🔥 Shenglin/视频下载"

    def download_batch(self, url, user_id, secret_key, media_types="video", max_videos=100):
        """批量提取并下载主页帖子"""
        # 固定参数：质量最高、自动翻页、不限页数、重试6次、延迟2秒
        quality_priority = "highest"
        auto_pagination = True
        max_pages = 9999  # 实际不限制，设置一个很大的值
        retry_times = 6
        retry_delay = 2
        # 1. 验证参数
        if not url or not url.strip():
            return ("", "错误: URL不能为空")
        if not user_id or not secret_key:
            return ("", "错误: 请配置userId和secretKey")

        # 统计信息
        total_posts = 0
        total_success = 0
        total_fail = 0
        all_downloaded_files = []
        pages_processed = 0
        username = "未知"
        user_id_str = ""
        output_folder = ""

        # 2. 循环提取和下载（支持分页）
        cursor = ""
        has_more = True

        while has_more and pages_processed < max_pages:
            pages_processed += 1
            print(f"\n{'='*50}")
            print(f"正在处理第 {pages_processed} 页...")
            print(f"{'='*50}")

            # 3. 提取当前页（带重试）
            page_data = self._fetch_page_with_retry(
                url, user_id, secret_key, cursor,
                retry_times, retry_delay
            )

            if page_data is None:
                # 提取失败
                status_msg = f"第 {pages_processed} 页提取失败，停止翻页\n"
                status_msg += self._generate_summary(
                    username, pages_processed, total_posts,
                    total_success, total_fail, output_folder, all_downloaded_files, max_videos
                )
                return (output_folder, status_msg)

            # 解析数据
            next_cursor = page_data.get("next_cursor", "")
            has_more = page_data.get("has_more", False)
            posts = page_data.get("posts", [])
            user_info = page_data.get("user", {})
            username = user_info.get("username", username)
            user_id_str = user_info.get("user_id", user_id_str)

            # 第一页时创建输出目录（使用 user_id 或 username 作为前缀）
            if pages_processed == 1:
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

            if not posts:
                print(f"第 {pages_processed} 页没有帖子")
                if not auto_pagination:
                    break
                continue

            # 限制每页帖子数
            posts = posts[:max_posts_per_page]
            total_posts += len(posts)

            # 4. 收集当前页的所有媒体
            page_medias = []
            for post in posts:
                post_id = post.get("id", "")
                post_text = post.get("text", "")
                medias = post.get("medias", [])

                for media in medias:
                    media["post_id"] = post_id
                    media["post_text"] = post_text
                    page_medias.append(media)

            # 过滤媒体类型
            if media_types != "all":
                page_medias = [m for m in page_medias if m.get("media_type") == media_types]

            if not page_medias:
                print(f"第 {pages_processed} 页没有{media_types}类型的媒体")
                if not auto_pagination:
                    break
                cursor = next_cursor
                continue

            # 检查是否已达到最大数量限制
            remaining_count = max_videos - total_success
            if remaining_count <= 0:
                print(f"已达到最大下载数量限制 ({max_videos}个)，停止下载")
                has_more = False
                break

            # 限制当前页下载数量，不超过剩余配额
            if len(page_medias) > remaining_count:
                print(f"当前页有 {len(page_medias)} 个媒体，但仅下载前 {remaining_count} 个（已达配额）")
                page_medias = page_medias[:remaining_count]

            # 5. 下载当前页的媒体
            success, fail, files = self._download_medias(
                page_medias, output_folder, username,
                quality_priority, retry_times, retry_delay
            )

            total_success += success
            total_fail += fail
            all_downloaded_files.extend(files)

            print(f"第 {pages_processed} 页下载完成: 成功 {success}个, 失败 {fail}个")

            # 6. 判断是否继续
            if not auto_pagination:
                break

            if not has_more:
                print("没有更多页了")
                break

            cursor = next_cursor

            # 页面间延迟，避免请求过快
            if has_more and pages_processed < max_pages:
                time.sleep(1)

        # 7. 生成最终报告
        status_msg = self._generate_summary(
            username, pages_processed, total_posts,
            total_success, total_fail, output_folder, all_downloaded_files, max_videos
        )

        return (output_folder, status_msg)

    def _fetch_page_with_retry(self, url, user_id, secret_key, cursor, retry_times, retry_delay):
        """带重试的页面提取"""
        for attempt in range(retry_times):
            try:
                params = {
                    "userId": user_id,
                    "secretKey": secret_key,
                    "url": url
                }

                if cursor:
                    params["cursor"] = cursor

                response = requests.post(
                    self.API_URL,
                    json=params,
                    timeout=60,
                    verify=False
                )

                result = response.json()

                if result.get("code") == 200 and result.get("succ"):
                    return result.get("data", {})
                elif result.get("code") == -30:
                    # 服务异常，重试
                    print(f"  服务异常(code: -30)，{retry_delay}秒后重试 ({attempt + 1}/{retry_times})")
                    time.sleep(retry_delay)
                    continue
                else:
                    # 其他错误
                    code = result.get("code", "unknown")
                    msg = result.get("msg", "未知错误")
                    print(f"  提取失败 (code: {code}): {msg}")
                    return None

            except requests.exceptions.Timeout:
                print(f"  请求超时，{retry_delay}秒后重试 ({attempt + 1}/{retry_times})")
                time.sleep(retry_delay)
                continue
            except Exception as e:
                print(f"  请求异常: {str(e)}，{retry_delay}秒后重试 ({attempt + 1}/{retry_times})")
                time.sleep(retry_delay)
                continue

        print(f"  重试 {retry_times} 次后仍然失败")
        return None

    def _download_medias(self, medias, output_folder, username, quality_priority, retry_times, retry_delay):
        """下载媒体列表"""
        success_count = 0
        fail_count = 0
        downloaded_files = []

        for idx, media in enumerate(medias):
            media_type = media.get("media_type", "unknown")
            post_id = media.get("post_id", "")

            # 获取下载URL
            download_url = self._get_download_url(media, quality_priority)

            if not download_url:
                fail_count += 1
                continue

            # 生成文件名
            ext = self._get_extension(download_url, media_type)
            filename = f"{username}_{post_id}_{media_type}_{int(time.time())}_{idx}{ext}"
            filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            filepath = os.path.join(output_folder, filename)

            # 下载（带重试）
            if self._download_file_with_retry(download_url, filepath, media, retry_times, retry_delay):
                success_count += 1
                downloaded_files.append(filename)
                print(f"  ✓ {filename}")
            else:
                fail_count += 1
                print(f"  ✗ {filename}")

        return success_count, fail_count, downloaded_files

    def _download_file_with_retry(self, url, filepath, media, retry_times, retry_delay):
        """带重试的文件下载"""
        headers = media.get("headers", {})
        if not headers:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

        for attempt in range(retry_times):
            try:
                resp = requests.get(
                    url,
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
                    return True
                else:
                    if attempt < retry_times - 1:
                        time.sleep(retry_delay)
                        continue
                    return False

            except Exception as e:
                if attempt < retry_times - 1:
                    time.sleep(retry_delay)
                    continue
                return False

        return False

    def _get_download_url(self, media, quality_priority):
        """获取下载URL"""
        if "formats" in media:
            formats = media["formats"]
            if quality_priority == "highest":
                selected = max(formats, key=lambda x: x.get("quality", 0))
            elif quality_priority == "lowest":
                selected = min(formats, key=lambda x: x.get("quality", 9999))
            else:
                sorted_formats = sorted(formats, key=lambda x: x.get("quality", 0))
                selected = sorted_formats[len(sorted_formats) // 2]

            if selected.get("separate") == 1:
                return selected.get("video_url")
            else:
                return selected.get("video_url")
        else:
            return media.get("resource_url")

    def _get_extension(self, url, media_type):
        """获取文件扩展名"""
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

    def _generate_summary(self, username, pages, total_posts, success, fail, output_folder, files, max_videos=None):
        """生成下载报告"""
        msg = f"{'='*50}\n"
        msg += f"🎉 批量下载完成!\n"
        msg += f"{'='*50}\n\n"
        msg += f"📊 统计信息:\n"
        msg += f"  作者: {username}\n"
        msg += f"  处理页数: {pages}页\n"
        msg += f"  处理帖子: {total_posts}个\n"
        if max_videos:
            msg += f"  数量限制: {max_videos}个\n"
        msg += f"  下载成功: {success}个\n"
        msg += f"  下载失败: {fail}个\n"
        msg += f"  成功率: {success/(success+fail)*100:.1f}%\n" if (success+fail) > 0 else f"  成功率: N/A\n"
        msg += f"\n📁 输出目录: {output_folder}\n"

        if files:
            msg += f"\n📥 已下载文件（显示前20个）:\n"
            for f in files[:20]:
                msg += f"  - {f}\n"
            if len(files) > 20:
                msg += f"  ... 还有 {len(files) - 20} 个文件\n"

        return msg


NODE_CLASS_MAPPINGS = {
    "哼哼猫-主页批量下载": BatchPostsDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "哼哼猫-主页批量下载": "哼哼猫-主页批量下载",
}
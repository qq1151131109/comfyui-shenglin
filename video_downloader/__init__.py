"""
哼哼猫视频下载节点
支持999+平台的视频/图片/音频提取和下载
"""

from .single_post_downloader import SinglePostDownloader
from .batch_posts_downloader import BatchPostsDownloader

NODE_CLASS_MAPPINGS = {
    "哼哼猫-单个帖子下载": SinglePostDownloader,
    "哼哼猫-主页批量下载": BatchPostsDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "哼哼猫-单个帖子下载": "哼哼猫-单个帖子下载",
    "哼哼猫-主页批量下载": "哼哼猫-主页批量下载",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
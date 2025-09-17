"""
ComfyUI Shenglin - 圣林的ComfyUI自定义节点集合
包含RunningHub API、MiniMax TTS、视频合成等完整工具链
"""

# RunningHub API模块
from .runninghub.batch_runninghub_text_to_image import RunningHubFluxTextToImage
from .runninghub.runninghub_qwen_advanced import RunningHubQwenAdvanced
from .runninghub.runninghub_qwen_text_to_image import RunningHubQwenTextToImage
from .runninghub.rh_execute_node import ExecuteNode
from .runninghub.rh_settings_node import SettingsNode
from .runninghub.rh_node_info_list import NodeInfoListNode
from .runninghub.rh_utils import AnyToStringNode, RH_Extract_Image_From_List, RH_Batch_Images_From_List
from .runninghub.rh_audio_uploader import RH_AudioUploader
from .runninghub.rh_video_uploader import RH_VideoUploader
from .runninghub.rh_image_uploader import ImageUploaderNode

# MiniMax TTS模块
from .minimax_tts.batch_minimax_tts import BatchMiniMaxTTSNode
from .minimax_tts.minimax_tts_dynamic import MiniMaxTTSDynamicNode
from .minimax_tts.batch_audio_preview import BatchAudioPreview

# 视频系统模块
from .video_system.enhanced_video_composer_v2 import EnhancedVideoComposerV2
from .video_system.video_preview import VideoPreview

# 节点映射
NODE_CLASS_MAPPINGS = {
    # RunningHub节点
    "RunningHubFluxTextToImage": RunningHubFluxTextToImage,
    "RunningHubQwenAdvanced": RunningHubQwenAdvanced,
    "RunningHubQwenTextToImage": RunningHubQwenTextToImage,
    "RHExecuteNode": ExecuteNode,
    "RHSettingsNode": SettingsNode,
    "RHNodeInfoListNode": NodeInfoListNode,
    "RHAnyToString": AnyToStringNode,
    "RHExtractImageFromList": RH_Extract_Image_From_List,
    "RHBatchImagesFromList": RH_Batch_Images_From_List,
    "RHAudioUploader": RH_AudioUploader,
    "RHVideoUploader": RH_VideoUploader,
    "RHImageUploader": ImageUploaderNode,

    # MiniMax TTS节点
    "BatchMiniMaxTTS": BatchMiniMaxTTSNode,
    "MiniMaxTTSDynamic": MiniMaxTTSDynamicNode,
    "BatchAudioPreview": BatchAudioPreview,

    # 视频系统节点
    "EnhancedVideoComposerV2": EnhancedVideoComposerV2,
    "VideoPreview": VideoPreview,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    # RunningHub节点
    "RunningHubFluxTextToImage": "🎨 RunningHub Flux文生图",
    "RunningHubQwenAdvanced": "🎨 RunningHub Qwen高级版",
    "RunningHubQwenTextToImage": "🎨 RunningHub Qwen文生图",
    "RHExecuteNode": "⚙️ RH执行节点",
    "RHSettingsNode": "🔧 RH设置节点",
    "RHNodeInfoListNode": "📋 RH节点信息列表",
    "RHAnyToString": "🔄 RH任意转字符串",
    "RHExtractImageFromList": "🖼️ RH提取图片",
    "RHBatchImagesFromList": "📦 RH批量图片",
    "RHAudioUploader": "🎵 RH音频上传",
    "RHVideoUploader": "🎬 RH视频上传",
    "RHImageUploader": "🖼️ RH图片上传",

    # MiniMax TTS节点
    "BatchMiniMaxTTS": "🎵 MiniMax批量TTS",
    "MiniMaxTTSDynamic": "🎤 MiniMax TTS (Dynamic)",
    "BatchAudioPreview": "🔊 批量音频预览",

    # 视频系统节点
    "EnhancedVideoComposerV2": "🎵 视频合成器",
    "VideoPreview": "📹 视频预览器",
}

# Web目录
WEB_DIRECTORY = "./video_system/web"

# 版本信息
__version__ = "1.0.0"
__author__ = "Shenglin"
__description__ = "圣林的ComfyUI自定义节点集合：RunningHub API集成、MiniMax TTS、视频合成工具链"

# ComfyUI必需的导出
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("🎉 ComfyUI Shenglin 节点集合加载完成!")
print(f"📝 RunningHub节点: 12个 | MiniMax TTS节点: 3个 | 视频系统节点: 2个")
print(f"🚀 总计: {len(NODE_CLASS_MAPPINGS)} 个自定义节点")
print("🎵 核心功能: 统一视频合成器 + 完整字体系统 + 音效库")
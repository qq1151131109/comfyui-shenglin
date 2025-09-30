"""
ComfyUI Shenglin - 圣林的ComfyUI自定义节点集合
包含RunningHub API、MiniMax TTS、视频合成等完整工具链
"""

# RunningHub API模块
from .runninghub.batch_runninghub_text_to_image import RunningHubFluxTextToImage
from .runninghub.runninghub_qwen_advanced import RunningHubQwenAdvanced
from .runninghub.runninghub_qwen_text_to_image import RunningHubQwenTextToImage
from .runninghub.runninghub_wan2_image_to_video import RunningHubWan2ImageToVideo
from .runninghub.runninghub_infinitetalk_video import RunningHubInfiniteTalkVideo
from .runninghub.batch_infinitetalk_video import BatchInfiniteTalkVideo
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
from .video_system.video_composer_from_videos import AIVideoComposer
from .video_system.video_preview import VideoPreview

# 批量工作流系统
try:
    from .batch_workflow_executor import BatchWorkflowExecutorNode
    BATCH_WORKFLOW_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import BatchWorkflowExecutorNode: {e}")
    BatchWorkflowExecutorNode = None
    BATCH_WORKFLOW_AVAILABLE = False

# RSS内容处理工具集
from .rss_tools import RSS_NODE_CLASS_MAPPINGS, RSS_NODE_DISPLAY_NAME_MAPPINGS

# Whisper语音识别和字幕生成工具集
from .whisper_tools import WHISPER_NODE_CLASS_MAPPINGS, WHISPER_NODE_DISPLAY_NAME_MAPPINGS

# 视频编辑工具集
from .video_editing_tools import VIDEO_EDITING_NODE_CLASS_MAPPINGS, VIDEO_EDITING_NODE_DISPLAY_NAME_MAPPINGS

# 视频下载工具集（哼哼猫）
from .video_downloader import NODE_CLASS_MAPPINGS as VIDEO_DOWNLOADER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DOWNLOADER_DISPLAY_MAPPINGS

# LLM大语言模型工具集（轻量级集成）
from .llm_tools import LLM_NODE_CLASS_MAPPINGS, LLM_NODE_DISPLAY_NAME_MAPPINGS, LLM_PARTY_AVAILABLE

# 节点映射
NODE_CLASS_MAPPINGS = {
    # RunningHub节点
    "RunningHubFluxTextToImage": RunningHubFluxTextToImage,
    "RunningHubQwenAdvanced": RunningHubQwenAdvanced,
    "RunningHubQwenTextToImage": RunningHubQwenTextToImage,
    "RunningHubWan2ImageToVideo": RunningHubWan2ImageToVideo,
    "RunningHubInfiniteTalkVideo": RunningHubInfiniteTalkVideo,
    "BatchInfiniteTalkVideo": BatchInfiniteTalkVideo,
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
    "AIVideoComposer": AIVideoComposer,
    "VideoPreview": VideoPreview,

    # 批量工作流节点
    **({"BatchWorkflowExecutorNode": BatchWorkflowExecutorNode} if BATCH_WORKFLOW_AVAILABLE else {}),

    # RSS内容处理节点
    **RSS_NODE_CLASS_MAPPINGS,

    # Whisper语音识别和字幕生成节点
    **WHISPER_NODE_CLASS_MAPPINGS,

    # 视频编辑工具节点
    **VIDEO_EDITING_NODE_CLASS_MAPPINGS,

    # 视频下载工具节点（哼哼猫）
    **VIDEO_DOWNLOADER_MAPPINGS,

    # LLM大语言模型节点（如果可用）
    **(LLM_NODE_CLASS_MAPPINGS if LLM_PARTY_AVAILABLE else {}),
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    # RunningHub节点
    "RunningHubFluxTextToImage": "🎨 RunningHub Flux文生图",
    "RunningHubQwenAdvanced": "🎨 RunningHub Qwen高级版",
    "RunningHubQwenTextToImage": "🎨 RunningHub Qwen文生图",
    "RunningHubWan2ImageToVideo": "🎬 RunningHub Wan2.2图生视频",
    "RunningHubInfiniteTalkVideo": "🎭 RunningHub InfiniteTalk数字人",
    "BatchInfiniteTalkVideo": "🎭 批量InfiniteTalk数字人",
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
    "EnhancedVideoComposerV2": "🖼️ 视频合成器-基于图片",
    "AIVideoComposer": "🎬 AI视频制作器",
    "VideoPreview": "📹 视频预览器",

    # 批量工作流节点
    **({"BatchWorkflowExecutorNode": "🔄 批量工作流执行器"} if BATCH_WORKFLOW_AVAILABLE else {}),

    # RSS内容处理节点
    **RSS_NODE_DISPLAY_NAME_MAPPINGS,

    # Whisper语音识别和字幕生成节点
    **WHISPER_NODE_DISPLAY_NAME_MAPPINGS,

    # 视频编辑工具节点
    **VIDEO_EDITING_NODE_DISPLAY_NAME_MAPPINGS,

    # 视频下载工具节点（哼哼猫）
    **VIDEO_DOWNLOADER_DISPLAY_MAPPINGS,

    # LLM大语言模型节点（如果可用）
    **(LLM_NODE_DISPLAY_NAME_MAPPINGS if LLM_PARTY_AVAILABLE else {}),
}

# Web目录
WEB_DIRECTORY = "./video_system/web"

# 版本信息
__version__ = "1.0.0"
__author__ = "Shenglin"
__description__ = "圣林的ComfyUI自定义节点集合：RunningHub API集成、MiniMax TTS、视频合成工具链、RSS内容处理套件、批量工作流执行器、Whisper语音识别、视频编辑工具、视频下载工具（哼哼猫）、LLM大语言模型"

# ComfyUI必需的导出
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("🎉 ComfyUI Shenglin 节点集合加载完成!")
llm_count = len(LLM_NODE_CLASS_MAPPINGS) if LLM_PARTY_AVAILABLE else 0
whisper_count = len(WHISPER_NODE_CLASS_MAPPINGS)
video_edit_count = len(VIDEO_EDITING_NODE_CLASS_MAPPINGS)
video_downloader_count = len(VIDEO_DOWNLOADER_MAPPINGS)

print(f"📝 RunningHub节点: 15个 | MiniMax TTS节点: 3个 | 视频系统节点: 3个 | 批量工作流节点: 1个")
print(f"📰 RSS处理节点: 11个 | 🎤 Whisper节点: {whisper_count}个 | ✂️ 视频编辑节点: {video_edit_count}个 | 📥 视频下载节点: {video_downloader_count}个")
if LLM_PARTY_AVAILABLE:
    print(f"🤖 LLM节点: {llm_count}个 (已集成)")
else:
    print("🤖 LLM节点: 未安装 (需要comfyui_LLM_party)")
print(f"🚀 总计: {len(NODE_CLASS_MAPPINGS)} 个自定义节点")
print("🎬 完整AI工作流生态: 图生视频 + 语音识别 + 智能分析 + 视频编辑 + 视频下载 + 大语言模型")

"""
ComfyUI Shenglin - 盛林的ComfyUI自定义节点集合
包含RunningHub API、MiniMax TTS、视频合成等完整工具链
"""

# RunningHub API模块
from .runninghub.batch_runninghub_text_to_image import RunningHubFluxTextToImage
from .runninghub.runninghub_qwen_advanced import RunningHubQwenAdvanced
from .runninghub.runninghub_qwen_text_to_image import RunningHubQwenTextToImage

# MiniMax TTS模块
from .minimax_tts.batch_minimax_tts import BatchMiniMaxTTS
from .minimax_tts.minimax_tts_dynamic import MiniMaxTTSDynamicNode
from .minimax_tts.batch_audio_preview import BatchAudioPreview

# 视频系统模块
from .video_system.video_composer import VideoComposer
from .video_system.video_preview import VideoPreview
from .video_system.story_timeline_builder import StoryTimelineBuilder
from .video_system.story_animation_processor import StoryAnimationProcessor
from .video_system.story_video_composer import StoryVideoComposer

# 节点映射
NODE_CLASS_MAPPINGS = {
    # RunningHub节点
    "RunningHubFluxTextToImage": RunningHubFluxTextToImage,
    "RunningHubQwenAdvanced": RunningHubQwenAdvanced,
    "RunningHubQwenTextToImage": RunningHubQwenTextToImage,

    # MiniMax TTS节点
    "BatchMiniMaxTTS": BatchMiniMaxTTS,
    "MiniMaxTTSDynamic": MiniMaxTTSDynamicNode,
    "BatchAudioPreview": BatchAudioPreview,

    # 视频系统节点
    "VideoComposer": VideoComposer,
    "VideoPreview": VideoPreview,
    "StoryTimelineBuilder": StoryTimelineBuilder,
    "StoryAnimationProcessor": StoryAnimationProcessor,
    "StoryVideoComposer": StoryVideoComposer,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    # RunningHub节点
    "RunningHubFluxTextToImage": "🎨 RunningHub Flux文生图",
    "RunningHubQwenAdvanced": "🎨 RunningHub Qwen高级版",
    "RunningHubQwenTextToImage": "🎨 RunningHub Qwen文生图",

    # MiniMax TTS节点
    "BatchMiniMaxTTS": "🎵 MiniMax批量TTS",
    "MiniMaxTTSDynamic": "🎤 MiniMax TTS (Dynamic)",
    "BatchAudioPreview": "🔊 批量音频预览",

    # 视频系统节点
    "VideoComposer": "🎬 视频合成器",
    "VideoPreview": "📹 视频预览器",
    "StoryTimelineBuilder": "⏱️ 故事时间轴构建器",
    "StoryAnimationProcessor": "🎭 故事动画处理器",
    "StoryVideoComposer": "🎞️ 故事视频合成器",
}

# Web目录
WEB_DIRECTORY = "./video_system/web"

# 版本信息
__version__ = "1.0.0"
__author__ = "Shenglin"
__description__ = "盛林的ComfyUI自定义节点集合：RunningHub API集成、MiniMax TTS、视频合成工具链"

# ComfyUI必需的导出
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("🎉 ComfyUI Shenglin 节点集合加载完成!")
print(f"📝 RunningHub节点: 3个 | MiniMax TTS节点: 3个 | 视频系统节点: 5个")
print(f"🚀 总计: {len(NODE_CLASS_MAPPINGS)} 个自定义节点")
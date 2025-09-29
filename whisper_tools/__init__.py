from .apply_whisper import ApplyWhisperNode
from .add_subtitles_to_frames import AddSubtitlesToFramesNode
from .add_subtitles_to_background import AddSubtitlesToBackgroundNode
from .resize_cropped_subtitles import ResizeCroppedSubtitlesNode

# 新增的WhisperX强制对齐和增强字幕节点
try:
    from .apply_whisperx_alignment import ApplyWhisperXAlignmentNode
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("⚠️ WhisperX依赖未安装，WhisperX强制对齐节点不可用")

try:
    from .add_subtitles_enhanced import AddSubtitlesEnhancedNode
    ENHANCED_SUBTITLES_AVAILABLE = True
except ImportError:
    ENHANCED_SUBTITLES_AVAILABLE = False
    print("⚠️ 增强字幕节点加载失败")

# Remotion V2 渲染节点（基于Enhanced架构）
try:
    from .remotion_captions_v2 import RenderRemotionCaptionsV2Node
    REMOTION_V2_AVAILABLE = True
except Exception as e:
    REMOTION_V2_AVAILABLE = False
    print(f"⚠️ Remotion V2 字幕渲染节点加载失败: {e}")

# 合并所有Whisper节点映射
WHISPER_NODE_CLASS_MAPPINGS = {
    "Apply Whisper" : ApplyWhisperNode,
    "Add Subtitles To Frames": AddSubtitlesToFramesNode,
    "Add Subtitles To Background": AddSubtitlesToBackgroundNode,
    "Resize Cropped Subtitles": ResizeCroppedSubtitlesNode
}

WHISPER_NODE_DISPLAY_NAME_MAPPINGS = {
     "Apply Whisper" : "🎤 Apply Whisper",
     "Add Subtitles To Frames": "📝 Add Subtitles To Frames",
     "Add Subtitles To Background": "🖼️ Add Subtitles To Background",
     "Resize Cropped Subtitles": "🔧 Resize Cropped Subtitles"
}

# 注册WhisperX强制对齐节点
if WHISPERX_AVAILABLE:
    WHISPER_NODE_CLASS_MAPPINGS["Apply WhisperX Alignment"] = ApplyWhisperXAlignmentNode
    WHISPER_NODE_DISPLAY_NAME_MAPPINGS["Apply WhisperX Alignment"] = "🎯 WhisperX 强制对齐"

# 注册增强字幕节点
if ENHANCED_SUBTITLES_AVAILABLE:
    WHISPER_NODE_CLASS_MAPPINGS["Add Subtitles Enhanced"] = AddSubtitlesEnhancedNode
    WHISPER_NODE_DISPLAY_NAME_MAPPINGS["Add Subtitles Enhanced"] = "🎨 Add Subtitles (Enhanced)"

# 注册 Remotion V2 节点
if REMOTION_V2_AVAILABLE:
    WHISPER_NODE_CLASS_MAPPINGS["Render Remotion Captions V2"] = RenderRemotionCaptionsV2Node
    WHISPER_NODE_DISPLAY_NAME_MAPPINGS["Render Remotion Captions V2"] = "🎬 Render TikTok Captions (Remotion V2)"

__all__ = ["WHISPER_NODE_CLASS_MAPPINGS", "WHISPER_NODE_DISPLAY_NAME_MAPPINGS"]
# RunningHub API Integration for ComfyUI
# 包含三个新开发的RunningHub节点

from .batch_runninghub_text_to_image import RunningHubFluxTextToImage
from .runninghub_qwen_advanced import RunningHubQwenAdvanced
from .runninghub_qwen_text_to_image import RunningHubQwenTextToImage

NODE_CLASS_MAPPINGS = {
    "RunningHubFluxTextToImage": RunningHubFluxTextToImage,
    "RunningHubQwenAdvanced": RunningHubQwenAdvanced,
    "RunningHubQwenTextToImage": RunningHubQwenTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubFluxTextToImage": "RunningHub Flux文生图",
    "RunningHubQwenAdvanced": "RunningHub Qwen高级版",
    "RunningHubQwenTextToImage": "RunningHub Qwen文生图",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
from .utils import tensor2pil, pil2tensor
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ResizeCroppedSubtitlesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "cropped_subtitles": ("IMAGE",),
                "original_frames": ("IMAGE",),
                "subtitle_coord": ("subtitle_coord",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "resize_cropped_subtitles"
    CATEGORY = "字幕"

    def resize_cropped_subtitles(self,cropped_subtitles, original_frames, subtitle_coord):

        pil_images_og = tensor2pil(original_frames)
        pil_images_cropped = tensor2pil(cropped_subtitles)
        final_images = []

        # 验证输入数据长度一致性
        if len(cropped_subtitles) != len(original_frames) or len(original_frames) != len(subtitle_coord):
            logger.warning(f"输入数据长度不一致: subtitles={len(cropped_subtitles)}, frames={len(original_frames)}, coords={len(subtitle_coord)}")

        width, height = pil_images_og[0].size

        for idx in range(len(pil_images_cropped)):
            frame = Image.new("RGB", (width, height), "black")
            frame.paste(pil_images_cropped[idx],(int(subtitle_coord[idx][0]),int(subtitle_coord[idx][1])))
            final_images.append(frame)

        return (pil2tensor(final_images),)

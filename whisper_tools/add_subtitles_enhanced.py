"""
ComfyUI 增强字幕添加节点
整合 comfy_add_subtitles 的高级样式功能
与现有 add_subtitles_to_frames 节点接口兼容，但功能更强大
"""

from PIL import ImageDraw, ImageFont, Image, ImageFilter
from .utils import tensor2pil, pil2tensor
import math
import os
import json
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


class SubtitleStyle:
    """字幕样式配置类"""
    
    def __init__(self, 
                 font_color: str = "white",
                 font_family: str = "Roboto-Regular.ttf", 
                 font_size: int = 100,
                 outline_color: str = "black",
                 outline_width: int = 3,
                 shadow_color: str = "black", 
                 shadow_offset: Tuple[int, int] = (2, 2),
                 background_color: str = None,
                 background_opacity: float = 0.7,
                 gradient_colors: List[str] = None,
                 text_align: str = "center"):
        self.font_color = font_color
        self.font_family = font_family
        self.font_size = font_size
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset
        self.background_color = background_color
        self.background_opacity = background_opacity
        self.gradient_colors = gradient_colors or []
        self.text_align = text_align


class AddSubtitlesEnhancedNode:
    """增强版字幕添加节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用字体
        try:
            available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith('.ttf')]
        except:
            available_fonts = ["Roboto-Regular.ttf"]
        
        return {
            "required": { 
                "images": ("IMAGE",),
                "alignment": ("whisper_alignment",),
                "video_fps": ("FLOAT", {
                    "default": 24.0,
                    "step": 1,
                    "display": "number"
                }),
                "preset": (["TikTok Boxed", "TikTok Outline", "Minimal"], {"default": "TikTok Boxed"}),
                "position_mode": (["bottom", "top"], {"default": "bottom"}),
                "safe_margin": ("INT", {"default": 80, "min": 0, "max": 300, "step": 5}),
            },
            "optional": {
                "font_family": (available_fonts, {"default": available_fonts[0] if available_fonts else "Roboto-Regular.ttf"}),
                "max_lines": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "text_case": (["none", "upper", "lower", "title"], {"default": "none"}),
                "font_mode": (["auto", "fixed", "scale"], {"default": "auto"}),
                "font_size": ("INT", {"default": 96, "min": 10, "max": 300, "step": 2}),
                "font_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "subtitle_coord", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "cropped_subtitles", "subtitle_coord", "render_log")
    FUNCTION = "add_subtitles_enhanced"
    CATEGORY = "字幕"

    def add_subtitles_enhanced(self, 
                             images,
                             alignment: List[Dict],
                             video_fps: float = 24.0,
                             preset: str = "TikTok Boxed",
                             position_mode: str = "bottom",
                             safe_margin: int = 80,
                             font_family: str = "Roboto-Regular.ttf",
                             max_lines: int = 2,
                             text_case: str = "none",
                             font_mode: str = "auto",
                             font_size: int = 96,
                             font_scale: float = 1.0) -> Tuple:
        """
        增强版字幕添加功能
        """
        log_messages = []
        
        try:
            # 转换图像格式
            pil_images = tensor2pil(images)
            log_messages.append(f"📷 处理图像数量: {len(pil_images)}")
            log_messages.append(f"🎬 视频帧率: {video_fps} fps")
            log_messages.append(f"🎯 对齐数据: {len(alignment)} 个片段")
            
            # 创建字幕样式（根据预设简化设置）
            style = self._apply_preset_simple(preset, font_family)
            
            # 加载字体
            try:
                # 计算最终字号（支持固定/倍率）
                final_font_size = style.font_size
                if font_mode == "fixed":
                    final_font_size = int(font_size)
                elif font_mode == "scale":
                    final_font_size = max(10, int(style.font_size * float(font_scale)))
                font_path = os.path.join(FONT_DIR, style.font_family)
                font = ImageFont.truetype(font_path, final_font_size)
                log_messages.append(f"🔤 字体加载成功: {style.font_family}")
            except Exception as e:
                # 使用默认字体作为后备
                font = ImageFont.load_default()
                log_messages.append(f"⚠️ 字体加载失败，使用默认字体: {e}")
            
            # 初始化输出列表
            pil_images_with_text = []
            cropped_pil_images_with_text = []
            pil_images_masks = []
            subtitle_coord = []
            
            # 处理空对齐情况
            if len(alignment) == 0:
                log_messages.append("⚠️ 无对齐数据，返回原始图像")
                return self._return_original_images(pil_images, log_messages)
            
            # 处理每一帧
            last_frame_no = 0
            processed_segments = 0
            
            for i, alignment_obj in enumerate(alignment):
                start_time = alignment_obj.get("start", 0.0)
                end_time = alignment_obj.get("end", 0.0) 
                text = alignment_obj.get("value", "").strip()
                
                if not text:
                    continue
                
                # 应用文字大小写转换
                text = self._apply_text_case(text, text_case)
                
                # CJK/一般文本清洗（统一空白）
                text = " ".join(text.split())
                
                start_frame_no = math.floor(start_time * video_fps)
                end_frame_no = math.floor(end_time * video_fps)
                
                # 确保帧数在有效范围内
                start_frame_no = max(0, min(start_frame_no, len(pil_images) - 1))
                end_frame_no = max(start_frame_no, min(end_frame_no, len(pil_images)))
                
                # 添加无字幕帧
                for frame_idx in range(last_frame_no, start_frame_no):
                    if frame_idx < len(pil_images):
                        self._add_empty_frame(
                            pil_images[frame_idx],
                            pil_images_with_text,
                            pil_images_masks,
                            cropped_pil_images_with_text,
                            subtitle_coord
                        )
                
                # 添加有字幕帧（简化渲染）
                for frame_idx in range(start_frame_no, end_frame_no):
                    if frame_idx < len(pil_images):
                        self._render_subtitle_frame_simple(
                            pil_images[frame_idx], text, font, style,
                            position_mode, safe_margin, max_lines,
                            font_mode,
                            pil_images_with_text, pil_images_masks,
                            cropped_pil_images_with_text, subtitle_coord,
                            1.0
                        )
                
                last_frame_no = end_frame_no
                processed_segments += 1
                
                log_messages.append(
                    f"✅ 片段 {i+1}: '{text[:20]}...' "
                    f"({start_time:.1f}s-{end_time:.1f}s, 帧{start_frame_no}-{end_frame_no})"
                )
            
            # 处理剩余帧
            for frame_idx in range(last_frame_no, len(pil_images)):
                self._add_empty_frame(
                    pil_images[frame_idx],
                    pil_images_with_text,
                    pil_images_masks, 
                    cropped_pil_images_with_text,
                    subtitle_coord
                )
            
            log_messages.append(f"🎉 字幕渲染完成!")
            log_messages.append(f"📊 统计信息:")
            log_messages.append(f"  - 处理片段数: {processed_segments}")
            log_messages.append(f"  - 总帧数: {len(pil_images_with_text)}")
            log_messages.append(f"  - 字幕帧数: {sum(1 for coord in subtitle_coord if coord != (0,0,0,0))}")
            
            # 转换回张量格式
            output_images = pil2tensor(pil_images_with_text)
            output_masks = pil2tensor([img.convert("L") for img in pil_images_masks])
            cropped_images = pil2tensor(cropped_pil_images_with_text)
            
            return (output_images, output_masks, cropped_images, subtitle_coord, "\n".join(log_messages))
            
        except Exception as e:
            error_msg = f"字幕渲染过程发生错误: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            log_messages.append(f"❌ {error_msg}")
            
            # 返回原始图像
            return self._return_original_images(pil_images, log_messages)
    
    def _parse_gradient_colors(self, gradient_str: str) -> List[str]:
        """解析渐变颜色字符串"""
        if not gradient_str.strip():
            return []
        
        try:
            colors = [color.strip() for color in gradient_str.split(',')]
            return [color for color in colors if color]
        except:
            return []
    
    def _apply_text_case(self, text: str, text_case: str) -> str:
        """应用文字大小写转换"""
        if text_case == "upper":
            return text.upper()
        elif text_case == "lower":
            return text.lower()
        elif text_case == "title":
            return text.title()
        return text

    def _render_subtitle_frame_simple(self, img, text, font, style,
                              position_mode, safe_margin, max_lines,
                              font_mode,
                              images_with_text, masks, cropped_images, coords,
                              alpha=1.0):
        """渲染单个字幕帧（精简版）"""
        img = img.convert("RGBA")
        width, height = img.size

        text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # 安全区与最大行宽（90% 安全区）
        safe_left = safe_margin
        safe_right = width - safe_margin
        safe_width = max(1, safe_right - safe_left)
        allowed_line_width = int(safe_width * 0.9)

        # 自动适配字体 + CJK换行（仅在auto时）
        if font_mode == "auto":
            font = self._auto_fit_font(font, text, style.font_family, allowed_line_width, max_lines)
        lines = self._wrap_text_smart(text, font, allowed_line_width)

        # 尺寸
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        total_height = sum(line_heights) + (len(lines) - 1) * int(line_heights[0] * (1.2 - 1))
        max_line_width = max(line_widths) if line_widths else 0

        # 位置（水平居中，上/下）
        start_x = (width - max_line_width) // 2
        start_y = max(height - safe_margin - total_height, safe_margin) if position_mode == "bottom" else safe_margin

        # 背景
        if style.background_color:
            bg_alpha = int(255 * style.background_opacity * alpha)
            bg_color = self._parse_color(style.background_color) + (bg_alpha,)
            padding = 20
            try:
                draw.rounded_rectangle([start_x - padding, start_y - padding, start_x + max_line_width + padding, start_y + total_height + padding], radius=12, fill=bg_color)
            except Exception:
                draw.rectangle([start_x - padding, start_y - padding, start_x + max_line_width + padding, start_y + total_height + padding], fill=bg_color)

        # 文本
        current_y = start_y
        text_coords = []
        for i, line in enumerate(lines):
            if not line.strip():
                current_y += int(line_heights[0] * 1.2)
                continue
            line_width = line_widths[i]
            line_x = start_x + (max_line_width - line_width) // 2

            if style.shadow_color and style.shadow_offset != (0, 0):
                shadow_x = line_x + style.shadow_offset[0]
                shadow_y = current_y + style.shadow_offset[1]
                shadow_color = self._parse_color(style.shadow_color) + (int(255 * alpha),)
                draw.text((shadow_x, shadow_y), line, font=font, fill=shadow_color)

            if style.outline_width > 0:
                outline_color = self._parse_color(style.outline_color) + (int(255 * alpha),)
                for dx in range(-style.outline_width, style.outline_width + 1):
                    for dy in range(-style.outline_width, style.outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((line_x + dx, current_y + dy), line, font=font, fill=outline_color)

            text_color = self._parse_color(style.font_color) + (int(255 * alpha),)
            draw.text((line_x, current_y), line, font=font, fill=text_color)

            bbox = draw.textbbox((line_x, current_y), line, font=font)
            text_coords.append(bbox)
            current_y += int(line_heights[i] * 1.2)

        result_img = Image.alpha_composite(img, text_layer).convert("RGB")
        images_with_text.append(result_img)

        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        for bbox in text_coords:
            mask_draw.rectangle(bbox, fill=255)
        masks.append(mask.convert("RGB"))

        if text_coords:
            min_x = min(bbox[0] for bbox in text_coords)
            min_y = min(bbox[1] for bbox in text_coords)
            max_x = max(bbox[2] for bbox in text_coords)
            max_y = max(bbox[3] for bbox in text_coords)
            cropped_subtitle = result_img.crop((min_x, min_y, max_x, max_y))
            cropped_images.append(cropped_subtitle)
            coords.append((min_x, min_y, max_x, max_y))
        else:
            small_black = Image.new('RGB', (1, 1), 'black')
            cropped_images.append(small_black)
            coords.append((0, 0, 0, 0))

    def _apply_preset_simple(self, preset: str, font_family: str) -> SubtitleStyle:
        preset_l = (preset or "").lower()
        if preset_l == "tiktok boxed":
            return SubtitleStyle(
                font_color="white",
                font_family=font_family,
                font_size=96,
                outline_color="black",
                outline_width=4,
                shadow_color="black",
                shadow_offset=(2, 2),
                background_color="#000000",
                background_opacity=0.75,
                gradient_colors=[],
                text_align="center",
            )
        if preset_l == "tiktok outline":
            return SubtitleStyle(
                font_color="white",
                font_family=font_family,
                font_size=96,
                outline_color="black",
                outline_width=6,
                shadow_color="black",
                shadow_offset=(1, 1),
                background_color=None,
                background_opacity=0.0,
                gradient_colors=[],
                text_align="center",
            )
        if preset_l == "minimal":
            return SubtitleStyle(
                font_color="white",
                font_family=font_family,
                font_size=90,
                outline_color="black",
                outline_width=2,
                shadow_color=None,
                shadow_offset=(0, 0),
                background_color=None,
                background_opacity=0.0,
                gradient_colors=[],
                text_align="center",
            )
        return SubtitleStyle(font_family=font_family)
    
    def _calculate_alpha(self, frame_time: float, start_time: float, end_time: float,
                        fade_in_duration: float, fade_out_duration: float) -> float:
        """计算帧的透明度（用于淡入淡出效果）"""
        duration = end_time - start_time
        relative_time = frame_time - start_time
        
        alpha = 1.0
        
        # 淡入效果
        if fade_in_duration > 0 and relative_time < fade_in_duration:
            alpha *= relative_time / fade_in_duration
        
        # 淡出效果  
        if fade_out_duration > 0 and relative_time > (duration - fade_out_duration):
            remaining_time = end_time - frame_time
            alpha *= remaining_time / fade_out_duration
        
        return max(0.0, min(1.0, alpha))
    
    def _add_empty_frame(self, img, images_with_text, masks, cropped_images, coords):
        """添加无字幕的帧"""
        img = img.convert("RGB")
        width, height = img.size
        
        images_with_text.append(img)
        
        # 创建黑色蒙版
        black_mask = Image.new('RGB', (width, height), 'black')
        masks.append(black_mask)
        
        # 创建最小尺寸的裁剪图像
        small_black = Image.new('RGB', (1, 1), 'black')
        cropped_images.append(small_black)
        
        coords.append((0, 0, 0, 0))
    
    def _render_subtitle_frame(self, img, text, font, style, x_pos, y_pos,
                              center_x, center_y, max_width, line_spacing, alpha,
                              position_mode, safe_margin, auto_fit, max_line_width_pct, max_lines,
                              background_padding, background_radius,
                              images_with_text, masks, cropped_images, coords):
        """渲染单个字幕帧"""
        img = img.convert("RGBA")
        width, height = img.size
        
        # 创建文字层
        text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # 安全区域宽度与最大行宽
        safe_left = safe_margin
        safe_right = width - safe_margin
        safe_width = max(1, safe_right - safe_left)
        allowed_line_width = int(safe_width * max_line_width_pct)

        # 处理换行（CJK 优化 + 指定最大宽度）
        effective_max_width = max(max_width, allowed_line_width) if max_width > 0 else allowed_line_width
        lines = self._wrap_text_smart(text, font, effective_max_width)

        # 自动适配：尝试缩放字体直到行宽/行数满足
        if auto_fit:
            font = self._auto_fit_font(font, text, style.font_family, effective_max_width, max_lines)
            # 重新计算行
            lines = self._wrap_text_smart(text, font, effective_max_width)
        
        # 计算文字总体尺寸
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        
        total_height = sum(line_heights) + (len(lines) - 1) * int(line_heights[0] * (line_spacing - 1))
        max_line_width = max(line_widths) if line_widths else 0
        
        # 计算起始位置
        if position_mode == "bottom":
            start_x = (width - max_line_width) // 2 if center_x else x_pos
            start_y = max(height - safe_margin - total_height, safe_margin)
        elif position_mode == "top":
            start_x = (width - max_line_width) // 2 if center_x else x_pos
            start_y = safe_margin
        else:
            # custom + 可选居中
            start_x = (width - max_line_width) // 2 if center_x else x_pos
            start_y = (height - total_height) // 2 if center_y else y_pos
        
        # 绘制背景（如果有）
        if style.background_color:
            bg_alpha = int(255 * style.background_opacity * alpha)
            bg_color = self._parse_color(style.background_color) + (bg_alpha,)
            
            # 计算背景矩形（兑现 background_padding + 圆角）
            padding = max(0, int(background_padding))
            bg_left = start_x - padding
            bg_top = start_y - padding
            bg_right = start_x + max_line_width + padding
            bg_bottom = start_y + total_height + padding
            
            try:
                draw.rounded_rectangle([bg_left, bg_top, bg_right, bg_bottom], radius=max(0, int(background_radius)), fill=bg_color)
            except Exception:
                # 旧版Pillow回退普通矩形
                draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_color)
        
        # 绘制每行文字
        current_y = start_y
        text_coords = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                current_y += int(line_heights[0] * line_spacing)
                continue
                
            # 计算该行的x位置
            line_width = line_widths[i]
            if style.text_align == "center":
                line_x = start_x + (max_line_width - line_width) // 2
            elif style.text_align == "right":
                line_x = start_x + max_line_width - line_width
            else:  # left
                line_x = start_x
            
            # 绘制阴影
            if style.shadow_color and style.shadow_offset != (0, 0):
                shadow_x = line_x + style.shadow_offset[0]
                shadow_y = current_y + style.shadow_offset[1]
                shadow_color = self._parse_color(style.shadow_color) + (int(255 * alpha),)
                draw.text((shadow_x, shadow_y), line, font=font, fill=shadow_color)
            
            # 绘制描边
            if style.outline_width > 0:
                outline_color = self._parse_color(style.outline_color) + (int(255 * alpha),)
                for dx in range(-style.outline_width, style.outline_width + 1):
                    for dy in range(-style.outline_width, style.outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((line_x + dx, current_y + dy), line, 
                                    font=font, fill=outline_color)
            
            # 绘制主文字
            text_color = self._parse_color(style.font_color) + (int(255 * alpha),)
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            
            # 记录坐标
            bbox = draw.textbbox((line_x, current_y), line, font=font)
            text_coords.append(bbox)
            
            current_y += int(line_heights[i] * line_spacing)
        
        # 合并图层
        result_img = Image.alpha_composite(img, text_layer)
        result_img = result_img.convert("RGB")
        
        images_with_text.append(result_img)
        
        # 创建蒙版
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # 在蒙版上绘制文字区域
        for bbox in text_coords:
            mask_draw.rectangle(bbox, fill=255)
        
        masks.append(mask.convert("RGB"))
        
        # 创建裁剪的字幕图像
        if text_coords:
            # 计算所有文字的边界框
            min_x = min(bbox[0] for bbox in text_coords)
            min_y = min(bbox[1] for bbox in text_coords) 
            max_x = max(bbox[2] for bbox in text_coords)
            max_y = max(bbox[3] for bbox in text_coords)
            
            # 裁剪字幕区域
            cropped_subtitle = result_img.crop((min_x, min_y, max_x, max_y))
            cropped_images.append(cropped_subtitle)
            coords.append((min_x, min_y, max_x, max_y))
        else:
            # 无文字时添加小黑图
            small_black = Image.new('RGB', (1, 1), 'black')
            cropped_images.append(small_black)
            coords.append((0, 0, 0, 0))
    
    def _wrap_text_smart(self, text: str, font, max_width: int) -> List[str]:
        """文字换行处理（支持CJK逐字换行）"""
        if max_width <= 0:
            return [text]
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        if self._contains_cjk(text) or (' ' not in text):
            # 逐字累积
            lines: List[str] = []
            current = ""
            for ch in text:
                test = current + ch
                bbox = draw.textbbox((0, 0), test, font=font)
                width = bbox[2] - bbox[0]
                if width <= max_width or current == "":
                    current = test
                else:
                    lines.append(current)
                    current = ch
            if current:
                lines.append(current)
            return lines
        else:
            # 单词换行
            words = text.split()
            lines: List[str] = []
            current_words: List[str] = []
            for word in words:
                test_line = " ".join(current_words + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                width = bbox[2] - bbox[0]
                if width <= max_width or not current_words:
                    current_words.append(word)
                else:
                    lines.append(" ".join(current_words))
                    current_words = [word]
            if current_words:
                lines.append(" ".join(current_words))
            return lines

    def _contains_cjk(self, text: str) -> bool:
        for ch in text:
            code = ord(ch)
            # 中日韩统一表意文字等主要区段
            if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x3040 <= code <= 0x30FF) or (0xAC00 <= code <= 0xD7AF):
                return True
        return False

    def _auto_fit_font(self, font, text: str, font_family: str, max_width: int, max_lines: int):
        """自动缩放字体，使行数与最大宽度达标"""
        try:
            # 通过现有font推断起始字号
            size = getattr(font, 'size', 100)
        except Exception:
            size = 100
        min_size = 20
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        while size > min_size:
            try:
                test_font = ImageFont.truetype(os.path.join(FONT_DIR, font_family), size)
            except Exception:
                test_font = font
            lines = self._wrap_text_smart(text, test_font, max_width)
            # 检查最大行宽
            max_line_w = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=test_font)
                max_line_w = max(max_line_w, bbox[2] - bbox[0])
            if max_line_w <= max_width and len(lines) <= max_lines:
                return test_font
            size -= 2
        return font
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """解析颜色字符串为RGB元组"""
        color_str = color_str.strip().lower()
        
        # 预定义颜色
        color_map = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }
        
        if color_str in color_map:
            return color_map[color_str]
        
        # 处理十六进制颜色
        if color_str.startswith('#'):
            try:
                hex_str = color_str[1:]
                if len(hex_str) == 6:
                    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                elif len(hex_str) == 3:
                    return tuple(int(hex_str[i]*2, 16) for i in range(3))
            except ValueError:
                pass
        
        # 处理RGB格式 rgb(255,255,255)
        if color_str.startswith('rgb(') and color_str.endswith(')'):
            try:
                rgb_str = color_str[4:-1]
                return tuple(int(x.strip()) for x in rgb_str.split(','))
            except ValueError:
                pass
        
        # 默认返回白色
        return (255, 255, 255)
    
    def _return_original_images(self, pil_images, log_messages):
        """返回原始图像（错误处理时使用）"""
        # 创建空蒙版和坐标
        width, height = pil_images[0].size if pil_images else (1, 1)
        black_masks = [Image.new('RGB', (width, height), 'black') for _ in pil_images]
        small_blacks = [Image.new('RGB', (1, 1), 'black') for _ in pil_images]
        empty_coords = [(0, 0, 0, 0) for _ in pil_images]
        
        output_images = pil2tensor(pil_images)
        output_masks = pil2tensor([img.convert("L") for img in black_masks])
        cropped_images = pil2tensor(small_blacks)
        
        return (output_images, output_masks, cropped_images, empty_coords, "\n".join(log_messages))


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AddSubtitlesEnhancedNode": AddSubtitlesEnhancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddSubtitlesEnhancedNode": "Add Subtitles (Enhanced)"
}


# 测试代码
if __name__ == "__main__":
    print("🎨 增强字幕添加节点测试")
    node = AddSubtitlesEnhancedNode()
    print("📋 输入类型:", node.INPUT_TYPES())
    print("🎯 返回类型:", node.RETURN_TYPES)
    print("📝 返回名称:", node.RETURN_NAMES)
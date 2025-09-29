"""
ComfyUI Remotion字幕渲染节点 V2
基于 Add Subtitles (Enhanced) 节点架构
使用 reference/remotion-subtitles 项目的样式系统
与🎯 WhisperX 强制对齐节点完美兼容
"""

import os
import json
import tempfile
import shutil
import subprocess
from typing import List, Dict, Tuple
import logging
from PIL import Image, ImageDraw

# 导入Enhanced节点的所有功能
from .add_subtitles_enhanced import AddSubtitlesEnhancedNode, SubtitleStyle
from .utils import tensor2pil, pil2tensor

logger = logging.getLogger(__name__)


class RenderRemotionCaptionsV2Node(AddSubtitlesEnhancedNode):
    """基于Enhanced节点架构的Remotion字幕渲染节点V2"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 继承Enhanced节点的基础接口
        base_types = super().INPUT_TYPES()
        
        # 添加Remotion特有的样式选项
        remotion_styles = [
            "neon_glow",        # 霓虹发光 (从reference项目)
        ]
        
        # 修改预设选项，添加Remotion样式
        base_types["required"]["preset"] = (remotion_styles, {"default": "neon_glow"})
        
        # 添加Remotion特有参数
        base_types["optional"]["switch_every_ms"] = ("INT", {
            "default": 1200, 
            "min": 100, 
            "max": 5000, 
            "step": 50,
            "tooltip": "TikTok样式字幕切换间隔(毫秒)"
        })
        base_types["optional"]["highlight_color"] = ("STRING", {
            "default": "#39E508",
            "tooltip": "高亮颜色(十六进制)"
        })
        
        return base_types

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "subtitle_coord", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "cropped_subtitles", "subtitle_coord", "render_log")
    FUNCTION = "render_remotion_captions_v2"
    CATEGORY = "字幕/Remotion"

    def render_remotion_captions_v2(self, 
                                   images,
                                   alignment: List[Dict],
                                   video_fps: float = 24.0,
                                   preset: str = "neon_glow",
                                   position_mode: str = "bottom",
                                   safe_margin: int = 80,
                                   font_family: str = "Roboto-Regular.ttf",
                                   max_lines: int = 2,
                                   text_case: str = "none",
                                   font_mode: str = "auto",
                                   font_size: int = 96,
                                   font_scale: float = 1.0,
                                   switch_every_ms: int = 1200,
                                   highlight_color: str = "#39E508") -> Tuple:
        """
        Remotion样式字幕渲染主函数
        """
        log_messages = []
        
        try:
            log_messages.append("🎬 Remotion字幕渲染器V2启动")
            log_messages.append(f"🎨 选择样式: {preset}")
            log_messages.append(f"⚡ 切换间隔: {switch_every_ms}ms")
            
            # 使用Remotion样式进行渲染
            log_messages.append(f"✨ 使用Remotion样式: {preset}")
            return self._render_with_remotion_style(
                images, alignment, video_fps, preset, position_mode, safe_margin,
                font_family, max_lines, text_case, font_mode, font_size, font_scale,
                switch_every_ms, highlight_color, log_messages
            )
                
        except Exception as e:
            error_msg = f"Remotion字幕渲染V2发生错误: {e}"
            logger.error(error_msg)
            log_messages.append(f"❌ {error_msg}")
            
            # 直接抛出异常，不隐藏问题
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Remotion V2渲染失败: {error_msg}") from e

    def _render_with_remotion_style(self, images, alignment, video_fps, preset, position_mode, 
                                  safe_margin, font_family, max_lines, text_case, font_mode, 
                                  font_size, font_scale, switch_every_ms, highlight_color, log_messages):
        """使用Remotion样式进行渲染"""
        
        try:
            # 检查Remotion项目目录
            remotion_dir = os.path.join(os.path.dirname(__file__), "remotion_v2")
            if not os.path.exists(remotion_dir):
                log_messages.append("⚠️ Remotion项目目录不存在，创建基础结构...")
                self._setup_remotion_project(remotion_dir, log_messages)
            
            # 转换图像格式
            pil_images = tensor2pil(images)
            log_messages.append(f"📷 处理图像数量: {len(pil_images)}")
            
            # 使用Remotion进行单帧渲染
            return self._render_frames_with_remotion(
                pil_images, alignment, video_fps, preset, switch_every_ms, 
                highlight_color, remotion_dir, log_messages
            )
            
        except Exception as e:
            error_msg = f"Remotion样式渲染失败: {e}"
            log_messages.append(f"❌ {error_msg}")
            logger.error(error_msg)
            
            # 直接抛出异常，暴露真实问题
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Remotion样式 {preset} 渲染失败: {error_msg}") from e

    def _setup_remotion_project(self, remotion_dir, log_messages):
        """设置基础Remotion项目结构"""
        try:
            os.makedirs(remotion_dir, exist_ok=True)
            
            # 创建基础package.json
            package_json = {
                "name": "comfyui-remotion-captions-v2",
                "version": "1.0.0",
                "dependencies": {
                    "@remotion/cli": "^4.0.0",
                    "@remotion/animation-utils": "^4.0.0", 
                    "react": "19.0.0",
                    "react-dom": "19.0.0",
                    "remotion": "^4.0.0"
                },
                "scripts": {
                    "build": "remotion bundle"
                }
            }
            
            package_json_path = os.path.join(remotion_dir, "package.json")
            with open(package_json_path, 'w', encoding='utf-8') as f:
                json.dump(package_json, f, indent=2)
                
            # 创建src目录和基础文件
            src_dir = os.path.join(remotion_dir, "src")
            os.makedirs(src_dir, exist_ok=True)
            
            # 创建基础index.ts
            index_ts = '''import { Composition } from "remotion";
import { NeonGlowVideo } from "./NeonGlowVideo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="NeonGlowVideo"
        component={NeonGlowVideo}
        width={720}
        height={1280}
        fps={30}
        durationInFrames={1}
      />
    </>
  );
};
'''
            
            with open(os.path.join(src_dir, "index.ts"), 'w', encoding='utf-8') as f:
                f.write(index_ts)
            
            # 创建基础NeonGlowVideo组件 
            neon_component = '''import React from "react";
import { AbsoluteFill } from "remotion";

export const NeonGlowVideo: React.FC<{
  text?: string;
  highlightColor?: string;
}> = ({ text = "Sample Text", highlightColor = "#39E508" }) => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: "black",
        justifyContent: "center",
        alignItems: "center",
        fontSize: 60,
        color: "white",
        textAlign: "center",
        textShadow: `0 0 20px ${highlightColor}, 0 0 40px ${highlightColor}`,
      }}
    >
      {text}
    </AbsoluteFill>
  );
};
'''
            
            with open(os.path.join(src_dir, "NeonGlowVideo.tsx"), 'w', encoding='utf-8') as f:
                f.write(neon_component)
                
            log_messages.append("✅ Remotion基础项目结构创建完成")
            
        except Exception as e:
            log_messages.append(f"❌ Remotion项目设置失败: {e}")
            raise

    def _render_frames_with_remotion(self, pil_images, alignment, video_fps, preset, 
                                   switch_every_ms, highlight_color, remotion_dir, log_messages):
        """使用Remotion渲染字幕帧（简化版实现）"""
        
        # 为了MVP版本，我们先用简单的霓虹发光效果代替完整的Remotion渲染
        log_messages.append("🎨 使用简化版霓虹发光效果 (MVP)")
        
        # 创建霓虹发光样式
        neon_style = SubtitleStyle(
            font_color="white",
            font_family="Roboto-Regular.ttf", 
            font_size=96,
            outline_color=highlight_color,
            outline_width=2,
            shadow_color=highlight_color,
            shadow_offset=(0, 0),  # 霓虹效果使用外发光而不是阴影
            background_color=None,
            background_opacity=0.0,
            text_align="center"
        )
        
        # 使用Enhanced节点的框架，但应用霓虹样式
        pil_images_with_text = []
        cropped_pil_images_with_text = []
        pil_images_masks = []
        subtitle_coord = []
        
        # 处理空对齐情况
        if len(alignment) == 0:
            log_messages.append("⚠️ 无对齐数据，返回原始图像")
            return self._return_original_images(pil_images, log_messages)
        
        # 加载字体
        try:
            from PIL import ImageFont
            font_path = os.path.join(os.path.dirname(__file__), "fonts", neon_style.font_family)
            font = ImageFont.truetype(font_path, neon_style.font_size)
            log_messages.append(f"🔤 字体加载成功: {neon_style.font_family}")
        except Exception as e:
            font = ImageFont.load_default()
            log_messages.append(f"⚠️ 字体加载失败，使用默认字体: {e}")
        
        # 处理每一帧
        import math
        last_frame_no = 0
        processed_segments = 0
        
        for i, alignment_obj in enumerate(alignment):
            start_time = alignment_obj.get("start", 0.0)
            end_time = alignment_obj.get("end", 0.0)
            text = alignment_obj.get("value", "").strip()
            
            if not text:
                continue
            
            # 应用TikTok风格的切换逻辑
            # 将长文本按switch_every_ms分割显示
            segment_duration = (end_time - start_time) * 1000  # 转换为毫秒
            num_switches = max(1, int(segment_duration / switch_every_ms))
            
            words = text.split()
            words_per_switch = max(1, len(words) // num_switches)
            
            for switch_idx in range(num_switches):
                switch_start_time = start_time + (switch_idx * switch_every_ms / 1000)
                switch_end_time = min(start_time + ((switch_idx + 1) * switch_every_ms / 1000), end_time)
                
                # 获取这个时间段要显示的文字
                word_start = switch_idx * words_per_switch
                word_end = min((switch_idx + 1) * words_per_switch, len(words))
                switch_text = " ".join(words[word_start:word_end]) if word_start < len(words) else ""
                
                if not switch_text:
                    continue
                
                start_frame_no = math.floor(switch_start_time * video_fps)
                end_frame_no = math.floor(switch_end_time * video_fps)
                
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
                
                # 添加有字幕帧（使用霓虹发光效果）
                for frame_idx in range(start_frame_no, end_frame_no):
                    if frame_idx < len(pil_images):
                        self._render_neon_glow_frame(
                            pil_images[frame_idx], switch_text, font, neon_style,
                            highlight_color,
                            pil_images_with_text, pil_images_masks,
                            cropped_pil_images_with_text, subtitle_coord
                        )
                
                last_frame_no = end_frame_no
            
            processed_segments += 1
            log_messages.append(f"✅ 片段 {i+1}: '{text[:20]}...' -> {num_switches}个切换")
        
        # 处理剩余帧
        for frame_idx in range(last_frame_no, len(pil_images)):
            self._add_empty_frame(
                pil_images[frame_idx],
                pil_images_with_text,
                pil_images_masks,
                cropped_pil_images_with_text,
                subtitle_coord
            )
        
        log_messages.append(f"🎉 霓虹发光字幕渲染完成!")
        log_messages.append(f"📊 统计信息:")
        log_messages.append(f"  - 处理片段数: {processed_segments}")
        log_messages.append(f"  - 总帧数: {len(pil_images_with_text)}")
        log_messages.append(f"  - 字幕帧数: {sum(1 for coord in subtitle_coord if coord != (0,0,0,0))}")
        
        # 转换回张量格式
        output_images = pil2tensor(pil_images_with_text)
        output_masks = pil2tensor([img.convert("L") for img in pil_images_masks])
        cropped_images = pil2tensor(cropped_pil_images_with_text)
        
        return (output_images, output_masks, cropped_images, subtitle_coord, "\n".join(log_messages))

    def _render_neon_glow_frame(self, img, text, font, style, highlight_color,
                              images_with_text, masks, cropped_images, coords):
        """渲染霓虹发光效果的单帧"""
        img = img.convert("RGBA")
        width, height = img.size

        text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # 计算文本位置（底部居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = height - text_height - 100  # 距离底部100px

        # 霓虹发光效果：多层外发光
        glow_color = self._parse_color(highlight_color)
        
        # 外层大发光（模糊效果）
        for offset in range(8, 0, -2):
            alpha = max(20, 80 - offset * 8)
            glow_with_alpha = glow_color + (alpha,)
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    if dx*dx + dy*dy <= offset*offset:
                        draw.text((x + dx, y + dy), text, font=font, fill=glow_with_alpha)

        # 内层强发光
        for offset in range(4, 0, -1):
            alpha = 120 - offset * 20
            glow_with_alpha = glow_color + (alpha,)
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    if abs(dx) + abs(dy) <= offset:
                        draw.text((x + dx, y + dy), text, font=font, fill=glow_with_alpha)

        # 主文字（白色）
        text_color = (255, 255, 255, 255)
        draw.text((x, y), text, font=font, fill=text_color)

        # 合并图层
        result_img = Image.alpha_composite(img, text_layer).convert("RGB")
        images_with_text.append(result_img)

        # 创建蒙版
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        text_bbox = (x, y, x + text_width, y + text_height)
        mask_draw.rectangle(text_bbox, fill=255)
        masks.append(mask.convert("RGB"))

        # 裁剪字幕区域
        cropped_subtitle = result_img.crop(text_bbox)
        cropped_images.append(cropped_subtitle)
        coords.append(text_bbox)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "RenderRemotionCaptionsV2Node": RenderRemotionCaptionsV2Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RenderRemotionCaptionsV2Node": "🎬 Render TikTok Captions (Remotion V2)"
}


# 测试代码
if __name__ == "__main__":
    print("🎬 Remotion字幕渲染器V2测试")
    try:
        # 在测试环境中，需要临时修改导入路径
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from add_subtitles_enhanced import AddSubtitlesEnhancedNode, SubtitleStyle
        from utils import tensor2pil, pil2tensor
        
        print("✅ 依赖模块导入成功")
        
        node = RenderRemotionCaptionsV2Node()
        print("📋 输入类型:", node.INPUT_TYPES())
        print("🎯 返回类型:", node.RETURN_TYPES)
        print("📝 返回名称:", node.RETURN_NAMES)
        print("🎉 节点测试通过!")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 这是正常的，因为模块需要在ComfyUI环境中运行")
    except Exception as e:
        print(f"❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
import json
import os
import shutil
import subprocess
import tempfile
from typing import Tuple, List, Dict
import logging

try:
    from .utils import tensor2pil, pil2tensor
except ImportError:
    try:
        # ComfyUI环境中的导入
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from utils import tensor2pil, pil2tensor
    except ImportError:
        # 最后的后备导入
        import torch
        from PIL import Image
        import numpy as np
        
        def tensor2pil(image):
            """将tensor转换为PIL图像列表"""
            if len(image.shape) == 4:  # batch dimension
                return [Image.fromarray(np.array(image[i].cpu().numpy() * 255, dtype=np.uint8)) for i in range(image.shape[0])]
            else:
                return [Image.fromarray(np.array(image.cpu().numpy() * 255, dtype=np.uint8))]
        
        def pil2tensor(images):
            """将PIL图像列表转换为tensor"""
            if isinstance(images, list):
                arrays = [np.array(img).astype(np.float32) / 255.0 for img in images]
                return torch.from_numpy(np.stack(arrays))
            else:
                array = np.array(images).astype(np.float32) / 255.0
                return torch.from_numpy(array).unsqueeze(0)

from PIL import Image

logger = logging.getLogger(__name__)


class TikTokCaptionsNode:
    @classmethod
    def INPUT_TYPES(cls):
        style_options = [
            "CaptionedVideo",          # 基础样式
            "MinimalStyle",            # 极简样式
            "NeonGlowStyle",           # 霓虹发光
            "RetroWaveStyle",          # 复古波浪
            "GlitchStyle",             # 故障风格
            "GlassmorphismStyle",      # 玻璃态
            "GradientRainbowStyle",    # 彩虹渐变
            "CartoonBubbleStyle",      # 卡通气泡
            "MetallicStyle",           # 金属质感
            "NeonBorderStyle",         # 霓虹边框
            "PixelArtStyle",           # 像素艺术
            "ElegantShadowStyle",      # 优雅阴影
            "Embossed3DStyle",         # 3D浮雕
            "FluidGradientStyle",      # 流体渐变
            "TechWireframeStyle",      # 科技线框
            "BouncyBallStyle",         # 弹跳球
            "SpinSpiralStyle",         # 旋转螺旋
            "ElasticZoomStyle",        # 弹性缩放
            "SwayBeatStyle",           # 摇摆节拍
            "ShakeImpactStyle",        # 震动冲击
            "FireFlameStyle",          # 火焰特效
            "WaterRippleStyle",        # 水波纹
            "ElectricLightningStyle",  # 电光特效
            "SmokeMistStyle",          # 烟雾迷雾
            "StarryParticleStyle",     # 星空粒子
            "QuantumTeleportStyle",    # 量子传送
            "HologramStyle",           # 全息投影
            "MagicSpellStyle",         # 魔法咒语
            "CyberpunkHackerStyle",    # 赛博朋克黑客
            "CosmicGalaxyStyle",       # 宇宙银河
            "SpaceTimeWarpStyle",      # 时空扭曲
            "DiamondCrystalStyle",     # 钻石水晶
            "AudioWaveStyle",          # 音频波形
            "LiquidMetalStyle",        # 液态金属
            "RainbowAuroraStyle",      # 彩虹极光
        ]
        
        return {
            "required": { 
                "images": ("IMAGE",),
                "alignment": ("whisper_alignment",),
                "video_fps": ("FLOAT", {
                    "default": 30.0,
                    "step": 1,
                    "display": "number"
                }),
                "style": (style_options, {"default": "MinimalStyle"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "render_log")
    FUNCTION = "render_tiktok_captions"
    CATEGORY = "字幕/TikTok"

    def render_tiktok_captions(self, 
                             images,
                             alignment: List[Dict],
                             video_fps: float = 30.0,
                             style: str = "MinimalStyle") -> Tuple:
        """
        使用Remotion渲染TikTok风格字幕
        """
        log_messages = []
        
        try:
            # 转换图像为视频文件
            pil_images = tensor2pil(images)
            log_messages.append(f"📷 输入图像: {len(pil_images)} 帧")
            log_messages.append(f"🎬 视频帧率: {video_fps} fps")
            log_messages.append(f"🎯 对齐数据: {len(alignment)} 个片段")
            log_messages.append(f"🎨 渲染样式: {style}")
            
            if len(pil_images) == 0:
                raise ValueError("没有输入图像")
            
            # 创建临时工作目录
            temp_dir = tempfile.mkdtemp(prefix="tiktok_render_")
            
            try:
                # 1. 将图像序列保存为视频文件
                input_video_path = self._create_video_from_images(pil_images, video_fps, temp_dir)
                log_messages.append(f"📹 输入视频: {os.path.basename(input_video_path)}")
                
                # 2. 转换字幕数据为Remotion格式
                captions_json_path = self._create_remotion_captions(alignment, temp_dir)
                log_messages.append(f"📝 字幕数据: {os.path.basename(captions_json_path)}")
                
                # 3. 使用Remotion渲染
                output_video_path = self._render_with_remotion(
                    input_video_path, captions_json_path, style, temp_dir
                )
                
                # 4. 将渲染结果转回图像序列
                output_images_tensor = self._extract_frames_from_video(output_video_path, video_fps)
                
                log_messages.append(f"✅ TikTok字幕渲染完成!")
                log_messages.append(f"📊 输出尺寸: {output_images_tensor.shape}")
                
                return (output_images_tensor, "\n".join(log_messages))
                
            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"TikTok字幕渲染失败: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            log_messages.append(f"❌ {error_msg}")
            
            # 返回原始图像
            return (images, "\n".join(log_messages))
    
    def _create_video_from_images(self, pil_images, fps, temp_dir):
        """将PIL图像序列转换为视频文件"""
        video_path = os.path.join(temp_dir, "input_video.mp4")
        
        # 保存所有帧到临时目录
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, img in enumerate(pil_images):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            img.save(frame_path)
        
        # 使用ffmpeg创建视频
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",  # 高质量
            video_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return video_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg视频创建失败: {e.stderr.decode()}")
    
    def _create_remotion_captions(self, alignment, temp_dir):
        """转换alignment数据为Remotion格式的字幕JSON"""
        captions_data = []
        
        for align_obj in alignment:
            start_ms = align_obj.get("start", 0.0) * 1000
            end_ms = align_obj.get("end", 0.0) * 1000
            text = align_obj.get("value", "").strip()
            
            if not text:
                continue
            
            captions_data.append({
                "startMs": int(start_ms),
                "endMs": int(end_ms),
                "timestampMs": int(start_ms),  # 使用开始时间作为时间戳
                "text": text,
                "confidence": align_obj.get("confidence", 1.0)
            })
        
        # 保存为JSON文件
        json_path = os.path.join(temp_dir, "video.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, ensure_ascii=False, indent=2)
        
        return json_path
    
    def _render_with_remotion(self, video_path, captions_path, style, temp_dir):
        """使用Remotion渲染字幕视频"""
        
        project_dir = os.path.join(os.path.dirname(__file__), "remotion")
        
        # 检查环境
        self._check_node()
        
        # 安装依赖
        self._ensure_node_dependencies(project_dir)
        
        # 将文件复制到Remotion项目的public目录
        project_public_dir = os.path.join(project_dir, "public")
        
        # 使用时间戳和进程ID确保文件名唯一，让字幕文件名与视频文件名匹配
        import time
        timestamp = int(time.time() * 1000)
        base_name = f"temp_video_{os.getpid()}_{timestamp}"
        temp_video_name = f"{base_name}.mp4"
        temp_captions_name = f"{base_name}.json"
        
        public_video_path = os.path.join(project_public_dir, temp_video_name)
        public_captions_path = os.path.join(project_public_dir, temp_captions_name)
        
        # 确保public目录存在
        os.makedirs(project_public_dir, exist_ok=True)
        
        # 复制文件并验证
        try:
            shutil.copy2(video_path, public_video_path)
            shutil.copy2(captions_path, public_captions_path)
            
            # 验证文件是否成功复制
            if not os.path.exists(public_video_path):
                raise RuntimeError(f"视频文件复制失败: {public_video_path}")
            if not os.path.exists(public_captions_path):
                raise RuntimeError(f"字幕文件复制失败: {public_captions_path}")
                
            logger.info(f"文件已复制到public目录: {temp_video_name}, {temp_captions_name}")
        except Exception as e:
            raise RuntimeError(f"文件复制失败: {e}")
        
        # 设置输出路径
        output_path = os.path.join(temp_dir, "output_with_captions.mp4")
        
        # 计算视频时长：根据字幕数据的最大时间点，加上缓冲时间
        max_end_time = 0
        if os.path.exists(captions_path):
            try:
                with open(captions_path, 'r', encoding='utf-8') as f:
                    captions_data = json.load(f)
                if captions_data:
                    max_end_time = max(caption.get('endMs', 0) for caption in captions_data)
            except:
                pass
        
        # 计算合适的视频时长：字幕时长 + 2秒缓冲，最少15秒
        duration_from_captions = (max_end_time / 1000.0) + 2.0 if max_end_time > 0 else 15.0
        video_duration = max(15.0, duration_from_captions)
        
        logger.info(f"计算视频时长: 字幕最大时间={max_end_time}ms, 设置时长={video_duration}秒")
        
        # 传递文件名和时长参数 - 使用 staticFile() 格式
        props = {
            "src": temp_video_name,
            "durationInSeconds": video_duration,
        }
        
        cmd = [
            "npx", "remotion", "render",
            "src/index.ts", 
            style,
            output_path,
            f"--props={json.dumps(props)}",
            "--timeout=600000",  # 10分钟超时
            "--concurrency=2",
            "--log=verbose"
        ]
    
        # 设置环境变量
        env = os.environ.copy()
        env.setdefault("NODE_OPTIONS", "--dns-result-order=ipv4first")
        
        try:
            logger.info(f"开始Remotion渲染，命令: {' '.join(cmd)}")
            logger.info(f"Props: {json.dumps(props, indent=2)}")
            
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=600  # 10分钟超时
            )
            
            logger.info(f"Remotion渲染成功完成")
            logger.info(f"输出文件: {output_path}")
            
            # 验证输出文件是否存在
            if not os.path.exists(output_path):
                raise RuntimeError(f"渲染完成但输出文件不存在: {output_path}")
            
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Remotion渲染超时 (10分钟)")
        except subprocess.CalledProcessError as e:
            logger.error(f"Remotion渲染失败:")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Remotion渲染失败:\n{e.stderr}")
        finally:
            # 清理临时文件 - 无论成功还是失败都清理
            try:
                if os.path.exists(public_video_path):
                    os.remove(public_video_path)
                    logger.info(f"已清理临时视频文件: {public_video_path}")
                if os.path.exists(public_captions_path):
                    os.remove(public_captions_path)
                    logger.info(f"已清理临时字幕文件: {public_captions_path}")
            except Exception as cleanup_error:
                logger.warning(f"清理临时文件时出错: {cleanup_error}")
    
    def _check_node(self):
        """检查Node.js环境"""
        try:
            subprocess.run(["node", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            subprocess.run(["npx", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception:
            raise RuntimeError("❌ Node.js / npx 未安装或不可用，请先安装 Node.js")

    def _ensure_node_dependencies(self, project_dir):
        """确保Node.js依赖已安装"""
        node_modules = os.path.join(project_dir, "node_modules")
        
        if not os.path.isdir(node_modules):
            try:
                subprocess.run(
                    ["npm", "install", "--silent", "--no-fund", "--no-audit"],
                    cwd=project_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as e:
                raise RuntimeError(f"npm install 失败: {e}")
    
    def _extract_frames_from_video(self, video_path, fps):
        """从视频文件提取帧并转换为张量"""
        temp_frames_dir = tempfile.mkdtemp(prefix="frames_extract_")
        
        try:
            # 使用ffmpeg提取帧
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"fps={fps}",
                os.path.join(temp_frames_dir, "frame_%06d.png")
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 读取所有帧
            frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
            
            if not frame_files:
                raise RuntimeError("未能从视频中提取帧")
            
            pil_images = []
            for frame_file in frame_files:
                frame_path = os.path.join(temp_frames_dir, frame_file)
                img = Image.open(frame_path).convert('RGB')
                pil_images.append(img)
            
            # 转换为张量
            return pil2tensor(pil_images)
            
        finally:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "TikTokCaptionsNode": TikTokCaptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TikTokCaptionsNode": "🎬 TikTok字幕渲染 (Remotion)"
}
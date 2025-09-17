"""
RunningHub Wan2.2图生视频节点
基于RunningHub高级API，使用工作流ID 1968308523518046210
支持从图片生成视频，使用Wan2.2 14B+LightX2V模型
"""

import asyncio
import aiohttp
import json
import time
import os
import ssl
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from PIL import Image
import io
import base64
import hashlib
import tempfile

class RunningHubWan2ImageToVideo:
    """
    RunningHub Wan2.2图生视频节点

    基于Wan2.2 14B+LightX2V模型，90秒81帧6步超极速图生视频
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_images": ("IMAGE", {
                    "tooltip": "输入图片（支持批量）"
                }),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "视频描述：一个美丽的场景慢慢展开",
                    "tooltip": "视频生成提示词，每行对应一张图片"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "RunningHub API密钥"
                }),

                # 音频输入（用于计算帧数）
                "audio_list": ("AUDIO_LIST", {
                    "tooltip": "音频列表，用于自动计算每个视频的帧数"
                }),

                # 视频参数
                "steps": ("INT", {
                    "default": 6,
                    "min": 4,
                    "max": 20,
                    "step": 1,
                    "tooltip": "推理步数（越高质量越好但越慢）"
                }),

                # 高级参数
                "cfg_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG缩放系数（提示词遵循度）"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "随机种子（-1为随机）"
                }),
                "motion_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "运动强度（0.1=静态，1.0=高动态）"
                }),

                # 批量处理
                "max_concurrent": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "tooltip": "最大并发数量"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量, 模糊, 变形",
                    "tooltip": "负面提示词（可选）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_paths", "info")
    FUNCTION = "generate_videos"
    CATEGORY = "🎨 RunningHub"

    def generate_videos(self, input_images, prompts, api_key, audio_list,
                       steps=6, cfg_scale=7.5, seed=-1, motion_strength=0.8,
                       max_concurrent=2, negative_prompt=""):
        """
        批量生成Wan2.2图生视频的主函数，根据音频时长自动计算帧数
        """
        # 固定参数配置
        workflow_id = "1968308523518046210"
        retry_count = 2
        retry_delay = 15
        timeout = 1200  # 视频生成需要更长时间
        fps = 16  # Wan2.2固定帧率为16fps

        if not api_key.strip():
            return ("", "错误: 请提供有效的RunningHub API密钥")

        # 获取音频时长列表
        try:
            audio_durations = self._get_audio_durations(audio_list)
            if not audio_durations:
                return ("", "错误: 无法获取音频时长信息")
        except Exception as e:
            return ("", f"错误: 音频处理失败 - {str(e)}")

        try:
            # 处理图片批次
            if len(input_images.shape) == 4:
                # 批量图片 (batch, height, width, channels)
                image_list = []
                for i in range(input_images.shape[0]):
                    image_array = input_images[i].cpu().numpy()
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    image_list.append(Image.fromarray(image_array))
            else:
                # 单张图片
                image_array = input_images.cpu().numpy()
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                image_list = [Image.fromarray(image_array)]

            # 处理提示词
            prompt_lines = [line.strip() for line in prompts.strip().split('\n') if line.strip()]
            if not prompt_lines:
                prompt_lines = ["视频描述：生成动态视频"]

            # 确保提示词数量与图片数量匹配
            while len(prompt_lines) < len(image_list):
                prompt_lines.append(prompt_lines[-1] if prompt_lines else "视频描述：生成动态视频")

            # 计算每个视频的帧数（根据对应音频时长）
            video_frames = []
            for i, (duration_sec, image) in enumerate(zip(audio_durations, image_list)):
                frames = max(25, min(200, int(duration_sec * fps)))  # 限制在25-200帧之间
                video_frames.append(frames)
                print(f"🎵 音频 {i+1}: {duration_sec:.1f}秒 → {frames}帧")

            print(f"🎬 开始批量生成视频...")
            print(f"📊 图片数量: {len(image_list)}")
            print(f"📝 提示词数量: {len(prompt_lines)}")
            print(f"🎵 音频数量: {len(audio_durations)}")
            print(f"⚙️ 视频参数: 16fps, {steps}步, 自动帧数")

            # 异步批量生成
            try:
                # 检查是否在现有的事件循环中
                loop = asyncio.get_running_loop()
                print("在现有事件循环中运行")

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._batch_generate_async(
                        image_list, prompt_lines, video_frames, api_key, workflow_id,
                        steps, cfg_scale, seed, motion_strength, max_concurrent,
                        retry_count, retry_delay, timeout, negative_prompt
                    ))
                    results = future.result()

            except RuntimeError as e:
                if "There is no current event loop" in str(e):
                    print("创建新的事件循环")
                    # 如果没有事件循环，创建一个新的
                    results = asyncio.run(
                        self._batch_generate_async(
                            image_list, prompt_lines, video_frames, api_key, workflow_id,
                            steps, cfg_scale, seed, motion_strength, max_concurrent,
                            retry_count, retry_delay, timeout, negative_prompt
                        )
                    )
                else:
                    # 如果循环未运行，直接运行
                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(
                        self._batch_generate_async(
                            image_list, prompt_lines, video_frames, api_key, workflow_id,
                            steps, cfg_scale, seed, motion_strength, max_concurrent,
                            retry_count, retry_delay, timeout, negative_prompt
                        )
                    )

            # 处理结果
            video_paths = []
            successful_count = 0
            failed_count = 0

            for i, result in enumerate(results):
                if result and result.get('success'):
                    video_path = result.get('video_path', '')
                    if video_path:
                        video_paths.append(video_path)
                        successful_count += 1
                    else:
                        video_paths.append("")
                        failed_count += 1
                        print(f"❌ 视频 {i+1} 路径为空")
                else:
                    video_paths.append("")
                    failed_count += 1
                    error_msg = result.get('error', '未知错误') if result else '生成失败'
                    print(f"❌ 视频 {i+1} 生成失败: {error_msg}")

            # 生成信息文本
            avg_duration = sum(audio_durations) / len(audio_durations) if audio_durations else 5.0
            avg_frames = sum(video_frames) / len(video_frames) if video_frames else 81
            info_lines = [
                f"RunningHub Wan2.2图生视频批量处理完成",
                f"工作流ID: {workflow_id}",
                f"成功生成: {successful_count} 个视频",
                f"失败数量: {failed_count} 个视频",
                f"视频参数: 16fps, 平均{avg_duration:.1f}秒, 平均{avg_frames:.0f}帧, {steps}步",
                f"运动强度: {motion_strength}, CFG: {cfg_scale}",
                f"处理时间: {len(image_list) * 2:.1f}分钟 (预估)"
            ]

            if successful_count > 0:
                info_lines.append("✅ 视频已保存到输出目录")

            info_text = "\n".join(info_lines)

            # 返回视频路径列表（用换行符分隔）
            video_paths_str = "\n".join(video_paths) if video_paths else ""

            return (video_paths_str, info_text)

        except Exception as e:
            error_msg = f"批量视频生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", error_msg)

    async def _batch_generate_async(self, image_list: List[Image.Image], prompt_list: List[str],
                                  frames_list: List[int], api_key: str, workflow_id: str, steps: int,
                                  cfg_scale: float, seed: int, motion_strength: float, max_concurrent: int,
                                  retry_count: int, retry_delay: int, timeout: int, negative_prompt: str) -> List[Optional[Dict]]:
        """
        异步批量生成视频
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []

        for index, (image, prompt, frames) in enumerate(zip(image_list, prompt_list, frames_list)):
            task = asyncio.create_task(
                self._generate_single_video_with_retry(
                    semaphore, index, image, prompt, frames, api_key, workflow_id,
                    steps, cfg_scale, seed, motion_strength, timeout, negative_prompt,
                    retry_count, retry_delay
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ 视频 {i+1} 生成异常: {result}")
                results[i] = None

        return results

    async def _generate_single_video_with_retry(self, semaphore: asyncio.Semaphore, index: int,
                                              image: Image.Image, prompt: str, frames: int, api_key: str,
                                              workflow_id: str, steps: int, cfg_scale: float, seed: int,
                                              motion_strength: float, timeout: int, negative_prompt: str,
                                              retry_count: int, retry_delay: int) -> Optional[Dict]:
        """
        带重试机制的单个视频生成
        """
        async with semaphore:
            for attempt in range(retry_count + 1):
                try:
                    if attempt > 0:
                        print(f"🎬 视频 {index+1} 重试 {attempt}/{retry_count}: {prompt[:30]}...")
                        await asyncio.sleep(retry_delay)
                    else:
                        print(f"🎬 开始生成视频 {index+1}: {prompt[:30]}...")

                    result = await self._generate_single_video(
                        image, prompt, frames, api_key, workflow_id, steps, cfg_scale,
                        seed, motion_strength, timeout, negative_prompt
                    )
                    print(f"✅ 视频 {index+1} 生成成功")
                    return result

                except Exception as e:
                    print(f"❌ 视频 {index+1} 第{attempt+1}次尝试失败: {e}")
                    if attempt == retry_count:
                        return {"success": False, "error": str(e)}
                    await asyncio.sleep(retry_delay)

            return {"success": False, "error": "所有重试均失败"}

    async def _generate_single_video(self, image: Image.Image, prompt: str, frames: int, api_key: str,
                                   workflow_id: str, steps: int, cfg_scale: float, seed: int,
                                   motion_strength: float, timeout: int, negative_prompt: str) -> Dict:
        """
        生成单个视频
        """
        # 转换图片为base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 处理种子
        if seed == -1:
            seed = int(time.time() * 1000) % 2147483647

        # 构建节点参数列表（根据Wan2.2工作流）
        node_list = [
            {
                "nodeId": 1,
                "type": "input_image",
                "data": {
                    "image": image_b64,
                    "format": "png"
                }
            },
            {
                "nodeId": 2,
                "type": "text_prompt",
                "data": {
                    "text": prompt
                }
            },
            {
                "nodeId": 3,
                "type": "video_params",
                "data": {
                    "frames": frames,  # 使用计算的帧数
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "motion_strength": motion_strength
                }
            }
        ]

        # 添加负面提示词（如果提供）
        if negative_prompt.strip():
            node_list.append({
                "nodeId": 4,
                "type": "negative_prompt",
                "data": {
                    "text": negative_prompt.strip()
                }
            })

        payload = {
            "apiKey": api_key,
            "workflowId": workflow_id,
            "nodeInfoList": node_list
        }

        # 创建SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(
                'https://api.runninghub.cn/api/v1/workflows/run',
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'ComfyUI-RunningHub/1.0'
                },
                json=payload
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败: {response.status} - {error_text}")

                result = await response.json()

                if not result.get('success', False):
                    error_msg = result.get('message', '未知错误')
                    raise Exception(f"视频生成失败: {error_msg}")

                # 获取视频结果
                video_data = result.get('data', {})
                video_url = video_data.get('videoUrl') or video_data.get('output_video')

                if not video_url:
                    raise Exception("视频URL为空")

                # 下载视频到本地
                video_path = await self._download_video(session, video_url, prompt)

                return {
                    "success": True,
                    "video_path": video_path,
                    "video_url": video_url,
                    "prompt": prompt,
                    "seed": seed
                }

    async def _download_video(self, session: aiohttp.ClientSession, video_url: str, prompt: str) -> str:
        """
        下载视频到本地
        """
        try:
            # 生成文件名
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_'))[:50]
            timestamp = int(time.time())
            filename = f"wan2_video_{timestamp}_{safe_prompt[:20]}.mp4"

            # 使用ComfyUI的输出目录
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except:
                output_dir = tempfile.gettempdir()

            video_path = os.path.join(output_dir, filename)

            async with session.get(video_url) as response:
                if response.status == 200:
                    with open(video_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

                    print(f"📹 视频已下载: {video_path}")
                    return video_path
                else:
                    raise Exception(f"下载失败: {response.status}")

        except Exception as e:
            print(f"❌ 视频下载失败: {e}")
            # 返回原始URL作为备选
            return video_url

    def _get_audio_durations(self, audio_list) -> List[float]:
        """
        获取音频列表中每个音频的时长（秒）
        """
        durations = []

        try:
            if not audio_list:
                raise ValueError("音频列表为空")

            for i, audio_path in enumerate(audio_list):
                if isinstance(audio_path, str) and os.path.exists(audio_path):
                    # 使用librosa或ffprobe获取音频时长
                    try:
                        import librosa
                        duration = librosa.get_duration(filename=audio_path)
                        durations.append(float(duration))
                        print(f"🎵 音频 {i+1}: {audio_path} - {duration:.2f}秒")
                    except ImportError:
                        # 如果没有librosa，尝试使用ffprobe
                        try:
                            import subprocess
                            import json

                            cmd = [
                                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                                '-show_format', audio_path
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                data = json.loads(result.stdout)
                                duration = float(data['format']['duration'])
                                durations.append(duration)
                                print(f"🎵 音频 {i+1}: {audio_path} - {duration:.2f}秒")
                            else:
                                print(f"⚠️ 无法获取音频时长: {audio_path}")
                                durations.append(5.0)  # 默认5秒
                        except Exception as e:
                            print(f"⚠️ FFprobe获取时长失败: {e}")
                            durations.append(5.0)  # 默认5秒
                else:
                    print(f"⚠️ 无效音频路径: {audio_path}")
                    durations.append(5.0)  # 默认5秒

        except Exception as e:
            print(f"❌ 获取音频时长失败: {e}")
            # 返回默认时长列表
            durations = [5.0] * len(audio_list)

        return durations

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RunningHubWan2ImageToVideo": RunningHubWan2ImageToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubWan2ImageToVideo": "🎬 RunningHub Wan2.2图生视频"
}
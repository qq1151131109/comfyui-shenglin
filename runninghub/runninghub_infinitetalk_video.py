"""
RunningHub InfiniteTalk数字人视频生成节点
基于RunningHub高级API，使用工作流ID 1960943620918579202
支持数字人视频生成，结合音频驱动人像说话
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
import folder_paths

class RunningHubInfiniteTalkVideo:
    """
    RunningHub InfiniteTalk数字人视频生成节点

    基于InfiniteTalk + Wan2.1模型，从人像视频和音频生成数字人说话视频
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 核心输入
                "character_video": ("STRING", {
                    "default": "",
                    "tooltip": "人像视频文件路径（支持mp4, avi, mov格式）"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "驱动语音音频"
                }),

                # API配置
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "RunningHub API密钥"
                }),

                # 视频参数
                "video_width": ("INT", {
                    "default": 480,
                    "min": 320,
                    "max": 1920,
                    "step": 16,
                    "tooltip": "输出视频宽度"
                }),
                "video_height": ("INT", {
                    "default": 832,
                    "min": 320,
                    "max": 1920,
                    "step": 16,
                    "tooltip": "输出视频高度"
                }),
                "fps": ("INT", {
                    "default": 25,
                    "min": 15,
                    "max": 60,
                    "tooltip": "输出帧率"
                }),

                # 生成参数
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "图中的角色在说话",
                    "tooltip": "正面提示词"
                }),
                "steps": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "tooltip": "采样步数"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "CFG引导强度"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "随机种子（-1为随机）"
                }),

                # 音频处理参数
                "audio_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "音频引导强度"
                }),
                "audio_crop_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 0.1,
                    "tooltip": "音频裁切起始时间(秒)"
                }),
                "audio_crop_end": ("FLOAT", {
                    "default": 60.0,
                    "min": 0.1,
                    "max": 300.0,
                    "step": 0.1,
                    "tooltip": "音频裁切结束时间(秒，0为全长)"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                    "tooltip": "负面提示词"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url", "task_info")
    FUNCTION = "generate_infinitetalk_video"
    CATEGORY = "🔥 Shenglin/RunningHub"
    DESCRIPTION = "基于RunningHub的InfiniteTalk数字人视频生成"

    def __init__(self):
        self.api_base = "https://www.runninghub.cn"
        self.workflow_id = "1960943620918579202"
        self.upload_timeout = 300  # 5分钟上传超时
        self.task_timeout = 1800   # 30分钟任务超时

    async def upload_file_async(self, session: aiohttp.ClientSession, file_path: str, api_key: str, file_type: str) -> Optional[str]:
        """异步上传文件到RunningHub"""
        try:
            print(f"🔄 开始上传{file_type}文件: {file_path}")

            # 获取上传URL
            upload_url_data = {
                "apiKey": api_key,
                "fileType": file_type
            }

            async with session.post(
                f"{self.api_base}/upload/openapi/url",
                json=upload_url_data,
                headers={"Host": "www.runninghub.cn"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    print(f"❌ 获取上传URL失败: {response.status}")
                    return None

                result = await response.json()
                if result.get("code") != 0:
                    print(f"❌ 获取上传URL失败: {result.get('msg', 'Unknown error')}")
                    return None

                upload_info = result.get("data", {})
                upload_url = upload_info.get("uploadUrl")
                file_key = upload_info.get("fileKey")

                if not upload_url or not file_key:
                    print("❌ 上传URL或文件密钥为空")
                    return None

            # 上传文件
            with open(file_path, 'rb') as file:
                file_data = file.read()

            async with session.put(
                upload_url,
                data=file_data,
                timeout=aiohttp.ClientTimeout(total=self.upload_timeout)
            ) as response:
                if response.status not in [200, 204]:
                    print(f"❌ 文件上传失败: {response.status}")
                    return None

            print(f"✅ {file_type}文件上传成功: {file_key}")
            return file_key

        except Exception as e:
            print(f"❌ 上传{file_type}文件异常: {str(e)}")
            return None

    async def create_task_async(self, session: aiohttp.ClientSession, api_key: str, node_info_list: List[Dict]) -> Optional[str]:
        """异步创建RunningHub任务"""
        try:
            task_data = {
                "apiKey": api_key,
                "workflowId": self.workflow_id,
                "nodeInfoList": node_info_list
            }

            print(f"🚀 创建InfiniteTalk数字人任务...")
            print(f"📝 节点参数数量: {len(node_info_list)}")

            async with session.post(
                f"{self.api_base}/task/openapi/create",
                json=task_data,
                headers={"Host": "www.runninghub.cn"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    print(f"❌ 创建任务失败: {response.status}")
                    return None

                result = await response.json()
                if result.get("code") != 0:
                    print(f"❌ 创建任务失败: {result.get('msg', 'Unknown error')}")
                    return None

                task_data = result.get("data", {})
                task_id = str(task_data.get("taskId", ""))
                task_status = task_data.get("taskStatus", "")

                print(f"✅ 任务创建成功!")
                print(f"📋 任务ID: {task_id}")
                print(f"📊 初始状态: {task_status}")

                return task_id

        except Exception as e:
            print(f"❌ 创建任务异常: {str(e)}")
            return None

    async def wait_for_completion_async(self, session: aiohttp.ClientSession, api_key: str, task_id: str) -> Optional[str]:
        """异步等待任务完成并获取结果"""
        try:
            print(f"⏳ 等待任务完成: {task_id}")
            start_time = time.time()

            while time.time() - start_time < self.task_timeout:
                # 查询任务状态
                status_data = {
                    "apiKey": api_key,
                    "taskId": task_id
                }

                async with session.post(
                    f"{self.api_base}/task/openapi/status",
                    json=status_data,
                    headers={"Host": "www.runninghub.cn"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        print(f"⚠️ 查询状态失败: {response.status}")
                        await asyncio.sleep(5)
                        continue

                    result = await response.json()
                    if result.get("code") != 0:
                        print(f"⚠️ 查询状态异常: {result.get('msg', 'Unknown error')}")
                        await asyncio.sleep(5)
                        continue

                    status = result.get("data", "")
                    print(f"📊 任务状态: {status}")

                    if status == "SUCCESS":
                        # 获取结果
                        async with session.post(
                            f"{self.api_base}/task/openapi/outputs",
                            json=status_data,
                            headers={"Host": "www.runninghub.cn"},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as result_response:
                            if result_response.status != 200:
                                print(f"❌ 获取结果失败: {result_response.status}")
                                return None

                            result_data = await result_response.json()
                            if result_data.get("code") != 0:
                                print(f"❌ 获取结果失败: {result_data.get('msg', 'Unknown error')}")
                                return None

                            outputs = result_data.get("data", [])
                            if outputs and len(outputs) > 0:
                                video_url = outputs[0].get("fileUrl", "")
                                if video_url:
                                    print(f"🎉 数字人视频生成完成!")
                                    print(f"🔗 视频URL: {video_url}")
                                    return video_url

                    elif status == "FAILED":
                        print(f"❌ 任务执行失败")
                        return None

                    elif status in ["QUEUED", "RUNNING"]:
                        await asyncio.sleep(10)  # 等待10秒后重试
                        continue

                    else:
                        print(f"⚠️ 未知状态: {status}")
                        await asyncio.sleep(5)

            print(f"⏰ 任务超时 ({self.task_timeout}秒)")
            return None

        except Exception as e:
            print(f"❌ 等待任务完成异常: {str(e)}")
            return None

    def generate_infinitetalk_video(
        self, character_video, audio, api_key, video_width, video_height, fps,
        prompt, steps, cfg_scale, seed, audio_scale, audio_crop_start, audio_crop_end,
        negative_prompt=""
    ):
        """生成InfiniteTalk数字人视频"""
        try:
            if not api_key:
                return ("", "❌ 错误：请提供RunningHub API密钥")

            if not character_video or not os.path.exists(character_video):
                return ("", "❌ 错误：请提供有效的人像视频文件路径")

            if audio is None:
                return ("", "❌ 错误：请提供音频数据")

            print("🎬 开始InfiniteTalk数字人视频生成...")
            print(f"📹 人像视频: {character_video}")
            print(f"🎵 音频数据: {type(audio)}")
            print(f"📐 输出尺寸: {video_width}x{video_height}")
            print(f"⚙️ 参数 - 步数:{steps}, CFG:{cfg_scale}, 种子:{seed}")

            # 运行异步生成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._generate_async(
                    character_video, audio, api_key, video_width, video_height, fps,
                    prompt, steps, cfg_scale, seed, audio_scale,
                    audio_crop_start, audio_crop_end, negative_prompt
                ))
                return result
            finally:
                loop.close()

        except Exception as e:
            print(f"❌ 生成过程异常: {str(e)}")
            return ("", f"❌ 生成失败: {str(e)}")

    async def _generate_async(
        self, character_video, audio, api_key, video_width, video_height, fps,
        prompt, steps, cfg_scale, seed, audio_scale,
        audio_crop_start, audio_crop_end, negative_prompt
    ):
        """异步生成逻辑"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.task_timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 1. 保存音频到临时文件
            temp_audio_path = None
            try:
                # 将ComfyUI音频保存为临时文件
                import torchaudio
                temp_audio_path = os.path.join(tempfile.gettempdir(), f"infinitetalk_audio_{int(time.time())}.wav")

                # 假设audio是字典格式 {"waveform": tensor, "sample_rate": int}
                if isinstance(audio, dict):
                    waveform = audio.get("waveform")
                    sample_rate = audio.get("sample_rate", 16000)
                else:
                    return ("", "❌ 错误：不支持的音频格式")

                # 保存音频文件
                torchaudio.save(temp_audio_path, waveform, sample_rate)
                print(f"💾 音频临时文件: {temp_audio_path}")

                # 2. 上传文件
                video_key = await self.upload_file_async(session, character_video, api_key, "video")
                if not video_key:
                    return ("", "❌ 视频文件上传失败")

                audio_key = await self.upload_file_async(session, temp_audio_path, api_key, "audio")
                if not audio_key:
                    return ("", "❌ 音频文件上传失败")

                # 3. 构建节点参数列表
                node_info_list = []

                # 设置音频文件 (节点125)
                node_info_list.append({
                    "nodeId": "125",
                    "fieldName": "audio",
                    "fieldValue": audio_key
                })

                # 设置视频文件 (节点302)
                node_info_list.append({
                    "nodeId": "302",
                    "fieldName": "video",
                    "fieldValue": video_key
                })

                # 设置正面提示词 (节点311)
                node_info_list.append({
                    "nodeId": "311",
                    "fieldName": "text",
                    "fieldValue": prompt
                })

                # 设置负面提示词 (节点241)
                if negative_prompt:
                    node_info_list.append({
                        "nodeId": "241",
                        "fieldName": "negative_prompt",
                        "fieldValue": negative_prompt
                    })

                # 设置视频尺寸 (节点304: 高度, 节点305: 宽度)
                node_info_list.append({
                    "nodeId": "304",
                    "fieldName": "value",
                    "fieldValue": video_height
                })
                node_info_list.append({
                    "nodeId": "305",
                    "fieldName": "value",
                    "fieldValue": video_width
                })

                # 设置音频裁切 (节点159)
                crop_start_str = f"0:{int(audio_crop_start//60):02d}:{int(audio_crop_start%60):02d}"
                crop_end_str = f"0:{int(audio_crop_end//60):02d}:{int(audio_crop_end%60):02d}"

                node_info_list.append({
                    "nodeId": "159",
                    "fieldName": "start_time",
                    "fieldValue": crop_start_str
                })
                node_info_list.append({
                    "nodeId": "159",
                    "fieldName": "end_time",
                    "fieldValue": crop_end_str
                })

                # 设置采样参数 (节点128)
                node_info_list.append({
                    "nodeId": "128",
                    "fieldName": "steps",
                    "fieldValue": steps
                })
                node_info_list.append({
                    "nodeId": "128",
                    "fieldName": "cfg",
                    "fieldValue": cfg_scale
                })

                if seed != -1:
                    node_info_list.append({
                        "nodeId": "128",
                        "fieldName": "seed",
                        "fieldValue": seed
                    })

                # 设置音频处理参数 (节点306)
                node_info_list.append({
                    "nodeId": "306",
                    "fieldName": "fps",
                    "fieldValue": fps
                })
                node_info_list.append({
                    "nodeId": "306",
                    "fieldName": "audio_scale",
                    "fieldValue": audio_scale
                })

                # 4. 创建任务
                task_id = await self.create_task_async(session, api_key, node_info_list)
                if not task_id:
                    return ("", "❌ 创建任务失败")

                # 5. 等待任务完成并获取结果
                video_url = await self.wait_for_completion_async(session, api_key, task_id)
                if not video_url:
                    return ("", f"❌ 任务执行失败或超时 (任务ID: {task_id})")

                # 6. 返回结果
                task_info = f"✅ InfiniteTalk数字人视频生成成功!\n📋 任务ID: {task_id}\n🎬 视频尺寸: {video_width}x{video_height}\n⚙️ 采样步数: {steps}\n🔗 视频URL: {video_url}"

                return (video_url, task_info)

            except Exception as e:
                print(f"❌ 异步生成异常: {str(e)}")
                return ("", f"❌ 生成异常: {str(e)}")

            finally:
                # 清理临时文件
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                        print(f"🗑️ 清理临时音频文件: {temp_audio_path}")
                    except Exception as e:
                        print(f"⚠️ 清理临时文件失败: {e}")

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "RunningHubInfiniteTalkVideo": RunningHubInfiniteTalkVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubInfiniteTalkVideo": "🎭 RunningHub InfiniteTalk数字人"
}
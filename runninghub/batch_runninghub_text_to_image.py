"""
批量RunningHub文生图节点
基于RunningHub API和工作流ID 1958005140101935106
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

class RunningHubFluxTextToImage:
    """
    RunningHub Flux文生图节点

    支持同时生成多张图片，基于指定的工作流ID，使用Flux模型
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "一个美丽的女孩\n一只可爱的猫咪\n壮观的山景",
                    "tooltip": "每行一个提示词，支持多行输入"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "RunningHub API密钥"
                }),
                "workflow_id": ("STRING", {
                    "default": "1958005140101935106",
                    "tooltip": "RunningHub工作流ID"
                }),
                "prompt_node_id": ("STRING", {
                    "default": "39",
                    "tooltip": "提示词节点ID"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "图片宽度"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "图片高度"
                }),
                "resolution_node_id": ("STRING", {
                    "default": "5",
                    "tooltip": "分辨率控制节点ID"
                }),
                "max_concurrent": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "tooltip": "最大并发生成数（建议1-2，避免队列满）"
                }),
                "retry_count": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "失败重试次数"
                }),
                "retry_delay": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 60,
                    "step": 5,
                    "tooltip": "重试延迟时间（秒）"
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 600,
                    "step": 10,
                    "tooltip": "单个任务超时时间（秒）"
                })
            },
            "optional": {
                "supports_custom_resolution": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否支持自定义分辨率"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate_batch_images"
    CATEGORY = "🎨 RunningHub"
    DESCRIPTION = "使用RunningHub Flux模型批量生成图片"

    def generate_batch_images(self, prompts: str, api_key: str, workflow_id: str,
                            prompt_node_id: str, width: int, height: int,
                            resolution_node_id: str, max_concurrent: int, retry_count: int,
                            retry_delay: int, timeout: int, supports_custom_resolution: bool = True):
        """
        批量生成图片的主函数
        """
        if not api_key.strip():
            raise ValueError("API密钥不能为空")

        if not workflow_id.strip():
            raise ValueError("工作流ID不能为空")

        # 解析提示词
        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        if not prompt_list:
            raise ValueError("至少需要一个有效的提示词")

        print(f"开始批量生成 {len(prompt_list)} 张图片...")

        # 运行异步生成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self._batch_generate_async(
                    prompt_list, api_key, workflow_id, prompt_node_id,
                    width, height, resolution_node_id, max_concurrent,
                    retry_count, retry_delay, timeout, supports_custom_resolution
                )
            )
        finally:
            loop.close()

        # 处理结果
        images = []
        info_lines = []

        for i, result in enumerate(results):
            if result is not None:
                images.append(result['image'])
                info_lines.append(f"图片 {i+1}: 成功生成 ({result['file_size']} bytes)")
            else:
                # 失败的任务用黑色图片占位
                placeholder = torch.zeros((height, width, 3), dtype=torch.float32)
                images.append(placeholder)
                info_lines.append(f"图片 {i+1}: 生成失败")

        # 合并图片张量
        if images:
            image_batch = torch.stack(images, dim=0)
        else:
            image_batch = torch.zeros((1, height, width, 3), dtype=torch.float32)

        info_text = "\n".join(info_lines)

        return (image_batch, info_text)

    async def _batch_generate_async(self, prompt_list: List[str], api_key: str,
                                  workflow_id: str, prompt_node_id: str,
                                  width: int, height: int, resolution_node_id: str,
                                  max_concurrent: int, retry_count: int, retry_delay: int,
                                  timeout: int, supports_custom_resolution: bool) -> List[Optional[Dict]]:
        """
        异步批量生成图片
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_single(prompt: str, index: int) -> Optional[Dict]:
            async with semaphore:
                for attempt in range(retry_count + 1):
                    try:
                        if attempt == 0:
                            print(f"开始生成图片 {index+1}: {prompt[:50]}...")
                        else:
                            print(f"图片 {index+1} 重试 {attempt}/{retry_count}: {prompt[:50]}...")

                        result = await self._generate_single_image(
                            prompt, api_key, workflow_id, prompt_node_id,
                            width, height, resolution_node_id, timeout,
                            supports_custom_resolution
                        )
                        print(f"图片 {index+1} 生成成功")
                        return result
                    except Exception as e:
                        error_msg = str(e)
                        print(f"图片 {index+1} 第 {attempt+1} 次尝试失败: {error_msg}")

                        # 如果是队列满或常见错误，等待后重试
                        if attempt < retry_count and ("timeout" in error_msg.lower() or "failed" in error_msg.lower()):
                            print(f"等待 {retry_delay} 秒后重试...")
                            await asyncio.sleep(retry_delay)
                        elif attempt < retry_count:
                            # 其他错误也等待一下再重试
                            await asyncio.sleep(retry_delay // 2)
                        else:
                            print(f"图片 {index+1} 所有重试都失败了")

                return None

        # 并发执行所有任务
        tasks = [generate_single(prompt, i) for i, prompt in enumerate(prompt_list)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    async def _generate_single_image(self, prompt: str, api_key: str, workflow_id: str,
                                   prompt_node_id: str, width: int, height: int,
                                   resolution_node_id: str, timeout: int,
                                   supports_custom_resolution: bool) -> Dict:
        """
        生成单张图片
        """
        # 构建节点参数列表
        node_list = [
            {
                "nodeId": prompt_node_id,
                "fieldName": "text",
                "fieldValue": prompt
            }
        ]

        # 如果支持自定义分辨率，添加分辨率控制
        if supports_custom_resolution and resolution_node_id:
            node_list.extend([
                {
                    "nodeId": resolution_node_id,
                    "fieldName": "width",
                    "fieldValue": width
                },
                {
                    "nodeId": resolution_node_id,
                    "fieldName": "height",
                    "fieldValue": height
                }
            ])

        payload = {
            "apiKey": api_key,
            "workflowId": workflow_id,
            "nodeInfoList": node_list
        }

        headers = {
            "Host": "www.runninghub.cn",
            "Content-Type": "application/json"
        }

        # 创建SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # 创建连接器
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            # 创建任务
            create_url = "https://www.runninghub.cn/task/openapi/create"
            async with session.post(create_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"创建任务失败 {response.status}: {error_text}")

                result = await response.json()
                if result.get('code') != 0:
                    error_msg = result.get('msg', '任务创建失败')
                    raise Exception(f"RunningHub API错误: {error_msg}")

                task_id = result.get('data', {}).get('taskId')
                if not task_id:
                    raise Exception("未返回任务ID")

            # 等待任务完成
            status_url = "https://www.runninghub.cn/task/openapi/status"
            outputs_url = "https://www.runninghub.cn/task/openapi/outputs"

            status_payload = {"taskId": task_id, "apiKey": api_key}

            start_time = time.time()
            while time.time() - start_time < timeout:
                await asyncio.sleep(2)

                try:
                    # 检查状态
                    async with session.post(status_url, json=status_payload, headers=headers) as status_response:
                        if status_response.status == 200:
                            status_result = await status_response.json()
                            if status_result.get('code') == 0:
                                task_status = status_result.get('data')

                                if task_status == 'SUCCESS':
                                    # 获取结果
                                    outputs_payload = {"taskId": task_id, "apiKey": api_key}
                                    async with session.post(outputs_url, json=outputs_payload, headers=headers) as outputs_response:
                                        if outputs_response.status == 200:
                                            outputs_result = await outputs_response.json()
                                            if outputs_result.get('code') == 0:
                                                outputs = outputs_result.get('data', [])

                                                # 查找图像URL
                                                for item in outputs:
                                                    image_url = None
                                                    if isinstance(item, dict) and 'fileUrl' in item:
                                                        image_url = item['fileUrl']
                                                    elif isinstance(item, str) and item.startswith('http'):
                                                        image_url = item

                                                    if image_url:
                                                        # 下载图像
                                                        async with session.get(image_url) as img_response:
                                                            if img_response.status == 200:
                                                                image_data = await img_response.read()

                                                                # 转换为tensor
                                                                pil_image = Image.open(io.BytesIO(image_data))
                                                                if pil_image.mode != 'RGB':
                                                                    pil_image = pil_image.convert('RGB')

                                                                # 转换为ComfyUI格式的tensor
                                                                image_np = np.array(pil_image).astype(np.float32) / 255.0
                                                                image_tensor = torch.from_numpy(image_np)

                                                                return {
                                                                    'image': image_tensor,
                                                                    'file_size': len(image_data),
                                                                    'task_id': task_id,
                                                                    'url': image_url
                                                                }

                                                raise Exception(f"输出中未找到图像URL: {outputs}")

                                elif task_status == 'FAILED':
                                    raise Exception(f"任务失败: {task_id}")

                except Exception as e:
                    # 如果是最终错误，抛出；否则继续轮询
                    if "任务失败" in str(e) or "输出中未找到" in str(e):
                        raise
                    continue

            raise Exception(f"任务超时: {task_id} ({timeout}秒)")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "RunningHubFluxTextToImage": RunningHubFluxTextToImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubFluxTextToImage": "RunningHub Flux文生图"
}
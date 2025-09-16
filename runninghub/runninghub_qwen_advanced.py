"""
RunningHub Qwen高级文生图节点
基于RunningHub高级API，使用工作流ID 1967888570176405505
支持自定义画面尺寸和更多高级参数
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

class RunningHubQwenAdvanced:
    """
    RunningHub Qwen高级文生图节点

    基于高级API和官方工作流，支持自定义尺寸和高级参数
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "一个美丽的女孩坐在花园里",
                    "tooltip": "每行一个提示词，支持多行批量输入"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "RunningHub API密钥"
                }),
                "width": ("INT", {
                    "default": 720,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "图片宽度（任意尺寸）"
                }),
                "height": ("INT", {
                    "default": 1280,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "图片高度（任意尺寸）"
                }),
                "max_concurrent": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "最大并发生成数"
                })
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低分辨率、模糊、失焦、像素化、噪点多、曝光异常、解剖错误、比例失调、多余肢体、不自然光影、饱和度异常、构图混乱、无关元素、纹理失真、版权违规、皮肤瑕疵、透视错误、悬浮物体、色彩违和、低对比度、边缘锯齿、过度对称、主体模糊",
                    "tooltip": "负向提示词 (高级选项)"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "采样步数 (高级选项)"
                }),
                "cfg": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG引导强度 (高级选项)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "随机种子，-1为随机 (高级选项)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate_advanced_images"
    CATEGORY = "🎨 Shenglin/RunningHub"
    DESCRIPTION = "使用RunningHub Qwen模型高级版本生成图片，支持自定义尺寸"

    def generate_advanced_images(self, prompts: str, api_key: str, width: int, height: int,
                                max_concurrent: int, negative_prompt: str = None,
                                steps: int = 20, cfg: float = 2.5, seed: int = -1):
        """
        批量生成Qwen高级图片的主函数
        """
        # 固定参数配置
        workflow_id = "1967888570176405505"
        retry_count = 2
        retry_delay = 10
        timeout = 600

        if not api_key.strip():
            raise ValueError("API密钥不能为空")

        # 解析提示词
        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        if not prompt_list:
            raise ValueError("至少需要一个有效的提示词")

        print(f"开始批量生成 {len(prompt_list)} 张Qwen高级图片 ({width}x{height})...")

        # 运行异步生成（使用现有的事件循环或创建新的）
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，使用 asyncio.run_coroutine_threadsafe
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._batch_generate_async(
                        prompt_list, api_key, workflow_id, width, height,
                        max_concurrent, retry_count, retry_delay, timeout,
                        negative_prompt, steps, cfg, seed
                    ))
                    results = future.result()
            else:
                # 如果循环未运行，直接运行
                results = loop.run_until_complete(
                    self._batch_generate_async(
                        prompt_list, api_key, workflow_id, width, height,
                        max_concurrent, retry_count, retry_delay, timeout,
                        negative_prompt, steps, cfg, seed
                    )
                )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            results = asyncio.run(
                self._batch_generate_async(
                    prompt_list, api_key, workflow_id, width, height,
                    max_concurrent, retry_count, retry_delay, timeout,
                    negative_prompt, steps, cfg, seed
                )
            )

        # 处理结果
        images = []
        info_lines = []
        total_images = 0

        for i, result in enumerate(results):
            # 安全的类型检查
            if isinstance(result, dict) and result.get('image') is not None:
                # 成功的任务
                images.append(result['image'])
                total_images += 1
                info_lines.append(f"✅ 提示词 {i+1}: 成功生成 ({width}x{height})")
            else:
                # 失败的任务用黑色图片占位
                placeholder = torch.zeros((height, width, 3), dtype=torch.float32)
                images.append(placeholder)
                total_images += 1

                # 显示具体错误信息
                if isinstance(result, dict) and result.get('error'):
                    error_info = result.get('error')
                    info_lines.append(f"❌ 提示词 {i+1}: 生成失败 - {error_info}")
                else:
                    info_lines.append(f"❌ 提示词 {i+1}: 生成失败，使用占位图片")

        # 合并图片张量
        if images:
            image_batch = torch.stack(images, dim=0)
        else:
            # 如果完全没有图片，创建一个默认占位符
            image_batch = torch.zeros((1, height, width, 3), dtype=torch.float32)

        info_text = f"总计生成 {total_images} 张图片 ({width}x{height})\n" + "\n".join(info_lines)

        return (image_batch, info_text)

    async def _batch_generate_async(self, prompt_list: List[str], api_key: str,
                                  workflow_id: str, width: int, height: int,
                                  max_concurrent: int, retry_count: int, retry_delay: int,
                                  timeout: int, negative_prompt: Optional[str],
                                  steps: int, cfg: float, seed: int) -> List[Optional[Dict]]:
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
                            prompt, api_key, workflow_id, width, height, timeout,
                            negative_prompt, steps, cfg, seed
                        )
                        print(f"图片 {index+1} 生成成功")
                        return result
                    except Exception as e:
                        error_msg = str(e)
                        print(f"❌ 图片 {index+1} 第 {attempt+1} 次尝试失败: {error_msg}")

                        if attempt < retry_count:
                            print(f"⏳ 等待 {retry_delay} 秒后重试...")
                            await asyncio.sleep(retry_delay)
                        else:
                            print(f"💀 图片 {index+1} 所有重试都失败了，最终错误: {error_msg}")
                            # 返回错误信息而不是None，便于调试
                            return {"error": error_msg, "index": index+1}

                return None

        # 并发执行所有任务
        tasks = [generate_single(prompt, i) for i, prompt in enumerate(prompt_list)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    async def _generate_single_image(self, prompt: str, api_key: str, workflow_id: str,
                                   width: int, height: int, timeout: int,
                                   negative_prompt: Optional[str], steps: int,
                                   cfg: float, seed: int) -> Dict:
        """
        生成单张高级图片
        """
        # 构建节点参数列表 (基于工作流JSON分析)
        node_list = [
            {
                "nodeId": "42",
                "fieldName": "text",
                "fieldValue": prompt
            },
            {
                "nodeId": "44",
                "fieldName": "width",
                "fieldValue": width
            },
            {
                "nodeId": "44",
                "fieldName": "height",
                "fieldValue": height
            },
            {
                "nodeId": "44",
                "fieldName": "batch_size",
                "fieldValue": 1
            },
            {
                "nodeId": "40",
                "fieldName": "steps",
                "fieldValue": steps
            },
            {
                "nodeId": "40",
                "fieldName": "cfg",
                "fieldValue": cfg
            }
        ]

        # 添加负向提示词（如果提供）
        if negative_prompt:
            node_list.append({
                "nodeId": "43",
                "fieldName": "text",
                "fieldValue": negative_prompt.strip()
            })

        # 添加种子（如果不是-1）
        if seed != -1:
            node_list.append({
                "nodeId": "40",
                "fieldName": "seed",
                "fieldValue": seed
            })

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
            # 发起高级任务
            create_url = "https://www.runninghub.cn/task/openapi/create"
            async with session.post(create_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"创建高级任务失败 {response.status}: {error_text}")

                result = await response.json()
                print(f"🔍 API响应: code={result.get('code')}, msg={result.get('msg')}")

                if result.get('code') != 0:
                    error_msg = result.get('msg', '任务创建失败')
                    print(f"❌ RunningHub API错误: {error_msg}")
                    raise Exception(f"RunningHub API错误: {error_msg}")

                data = result.get('data', {})
                task_id = data.get('taskId')
                print(f"📋 获取任务ID: {task_id}")

                if not task_id:
                    print(f"❌ API返回数据异常: {data}")
                    raise Exception("未返回任务ID")

            # 等待任务完成
            return await self._wait_for_completion_polling(session, task_id, api_key, timeout)

    async def _wait_for_completion_polling(self, session: aiohttp.ClientSession,
                                         task_id: str, api_key: str, timeout: int) -> Dict:
        """
        使用轮询方式等待任务完成
        """
        status_url = "https://www.runninghub.cn/task/openapi/status"
        outputs_url = "https://www.runninghub.cn/task/openapi/outputs"

        headers = {
            "Host": "www.runninghub.cn",
            "Content-Type": "application/json"
        }

        status_payload = {"taskId": task_id, "apiKey": api_key}

        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(3)

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

                                                            # 保持原始生成的尺寸，不强制resize
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
                                # 获取失败原因
                                try:
                                    outputs_payload = {"taskId": task_id, "apiKey": api_key}
                                    async with session.post(outputs_url, json=outputs_payload, headers=headers) as outputs_response:
                                        if outputs_response.status == 200:
                                            outputs_result = await outputs_response.json()
                                            if outputs_result.get('code') == 805:  # 任务失败
                                                fail_data = outputs_result.get('data', {}).get('failedReason', {})
                                                fail_msg = fail_data.get('exception_message', 'Unknown error')
                                                raise Exception(f"任务失败: {fail_msg}")
                                except:
                                    pass
                                raise Exception(f"高级任务失败: {task_id}")

                            elif task_status in ['RUNNING', 'QUEUED']:
                                # 继续等待
                                continue

            except Exception as e:
                # 如果是最终错误，抛出；否则继续轮询
                if "任务失败" in str(e) or "输出中未找到" in str(e):
                    raise
                continue

        raise Exception(f"高级任务超时: {task_id} ({timeout}秒)")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "RunningHubQwenAdvanced": RunningHubQwenAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubQwenAdvanced": "RunningHub Qwen高级版"
}
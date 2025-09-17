"""
RunningHub Qwen文生图节点
基于RunningHub快捷创作API，使用Qwen模型生成图片
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

class RunningHubQwenTextToImage:
    """
    RunningHub Qwen文生图节点

    基于快捷创作API，使用Qwen模型生成图片
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
                "aspect_ratio": ("COMBO", {
                    "options": ["正方形", "横屏", "竖屏"],
                    "default": "正方形",
                    "tooltip": "图片比例"
                }),
                "text_input_method": ("COMBO", {
                    "options": ["手写输入", "润色输入"],
                    "default": "手写输入",
                    "tooltip": "文本输入方式"
                }),
                "max_concurrent": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "最大并发生成数"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate_qwen_images"
    CATEGORY = "🔥 Shenglin/图像生成"
    DESCRIPTION = "使用RunningHub Qwen模型批量生成图片"

    def generate_qwen_images(self, prompts: str, api_key: str, aspect_ratio: str,
                           text_input_method: str, max_concurrent: int):
        """
        批量生成Qwen图片的主函数
        """
        # 固定参数配置
        batch_size = 1
        webapp_id = "1959831828515467265"
        quick_create_code = "006"
        retry_count = 2
        retry_delay = 10
        timeout = 600

        if not api_key.strip():
            raise ValueError("API密钥不能为空")

        if not webapp_id.strip():
            raise ValueError("WebApp ID不能为空")

        # 解析提示词
        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        if not prompt_list:
            raise ValueError("至少需要一个有效的提示词")

        print(f"开始批量生成 {len(prompt_list)} 张Qwen图片...")

        # 运行异步生成（使用现有的事件循环或创建新的）
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，使用 asyncio.run_coroutine_threadsafe
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._batch_generate_async(
                        prompt_list, api_key, webapp_id, quick_create_code,
                        batch_size, aspect_ratio, text_input_method,
                        max_concurrent, retry_count, retry_delay, timeout
                    ))
                    results = future.result()
            else:
                # 如果循环未运行，直接运行
                results = loop.run_until_complete(
                    self._batch_generate_async(
                        prompt_list, api_key, webapp_id, quick_create_code,
                        batch_size, aspect_ratio, text_input_method,
                        max_concurrent, retry_count, retry_delay, timeout
                    )
                )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            results = asyncio.run(
                self._batch_generate_async(
                    prompt_list, api_key, webapp_id, quick_create_code,
                    batch_size, aspect_ratio, text_input_method,
                    max_concurrent, retry_count, retry_delay, timeout
                )
            )

        # 处理结果
        images = []
        info_lines = []
        total_images = 0

        for i, result in enumerate(results):
            if result is not None and result.get('images'):
                # 成功的任务
                result_images = result['images']
                for j, img in enumerate(result_images):
                    images.append(img)
                    total_images += 1
                info_lines.append(f"提示词 {i+1}: 成功生成 {len(result_images)} 张图片")
            else:
                # 失败的任务用黑色图片占位
                for _ in range(batch_size):
                    # 使用默认分辨率 1024x1024
                    placeholder = torch.zeros((1024, 1024, 3), dtype=torch.float32)
                    images.append(placeholder)
                    total_images += 1
                info_lines.append(f"提示词 {i+1}: 生成失败，使用占位图片")

        # 合并图片张量
        if images:
            image_batch = torch.stack(images, dim=0)
        else:
            # 如果完全没有图片，创建一个默认占位符
            image_batch = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)

        info_text = f"总计生成 {total_images} 张图片\n" + "\n".join(info_lines)

        return (image_batch, info_text)


    async def _batch_generate_async(self, prompt_list: List[str], api_key: str,
                                  webapp_id: str, quick_create_code: str,
                                  batch_size: int, aspect_ratio: str,
                                  text_input_method: str, max_concurrent: int,
                                  retry_count: int, retry_delay: int,
                                  timeout: int) -> List[Optional[Dict]]:
        """
        异步批量生成图片
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_single(prompt: str, index: int) -> Optional[Dict]:
            async with semaphore:
                for attempt in range(retry_count + 1):
                    try:
                        if attempt == 0:
                            print(f"开始生成第 {index+1} 组图片: {prompt[:50]}...")
                        else:
                            print(f"第 {index+1} 组图片重试 {attempt}/{retry_count}: {prompt[:50]}...")

                        result = await self._generate_single_qwen_image(
                            prompt, api_key, webapp_id, quick_create_code,
                            batch_size, aspect_ratio, text_input_method, timeout
                        )
                        print(f"第 {index+1} 组图片生成成功")
                        return result
                    except Exception as e:
                        error_msg = str(e)
                        print(f"第 {index+1} 组图片第 {attempt+1} 次尝试失败: {error_msg}")

                        # 如果是队列满或特定错误，等待后重试
                        if attempt < retry_count and ("TASK_QUEUE_MAXED" in error_msg or "未返回WebSocket URL" in error_msg):
                            print(f"等待 {retry_delay} 秒后重试...")
                            await asyncio.sleep(retry_delay)
                        elif attempt < retry_count:
                            # 其他错误也等待一下再重试
                            await asyncio.sleep(retry_delay // 2)
                        else:
                            print(f"第 {index+1} 组图片所有重试都失败了")

                return None

        # 并发执行所有任务
        tasks = [generate_single(prompt, i) for i, prompt in enumerate(prompt_list)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    async def _generate_single_qwen_image(self, prompt: str, api_key: str,
                                        webapp_id: str, quick_create_code: str,
                                        batch_size: int, aspect_ratio: str,
                                        text_input_method: str, timeout: int) -> Dict:
        """
        生成单组Qwen图片
        """
        # 映射比例设置 (根据实际测试结果调整)
        aspect_ratio_mapping = {
            "正方形": "1",    # 正方形
            "横屏": "3",      # 横屏 (可能原来搞反了)
            "竖屏": "2"       # 竖屏 (可能原来搞反了)
        }

        # 映射文本输入方式
        text_method_mapping = {
            "手写输入": "1",
            "润色输入": "2"
        }

        # 构建节点参数列表
        node_list = [
            {
                "nodeId": "889",
                "nodeName": "EmptyLatentImage",
                "fieldName": "batch_size",
                "fieldType": "INT",
                "fieldValue": str(batch_size),
                "description": "生成张数"
            },
            {
                "nodeId": "887",
                "nodeName": "ImpactSwitch",
                "fieldName": "select",
                "fieldType": "SWITCH",
                "fieldValue": aspect_ratio_mapping.get(aspect_ratio, "1"),
                "description": "设置比例"
            },
            {
                "nodeId": "923",
                "nodeName": "easy anythingIndexSwitch",
                "fieldName": "index",
                "fieldType": "SWITCH",
                "fieldValue": text_method_mapping.get(text_input_method, "1"),
                "description": "文本输入方式"
            },
            {
                "nodeId": "876",
                "nodeName": "JjkText",
                "fieldName": "text",
                "fieldType": "STRING",
                "fieldValue": prompt,
                "description": "手写/润色文本输入框"
            }
        ]

        payload = {
            "webappId": webapp_id,
            "apiKey": api_key,
            "quickCreateCode": quick_create_code,
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
            # 发起快捷创作任务
            create_url = "https://www.runninghub.cn/task/openapi/quick-ai-app/run"
            async with session.post(create_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"创建快捷创作任务失败 {response.status}: {error_text}")

                result = await response.json()
                if result.get('code') != 0:
                    error_msg = result.get('msg', '任务创建失败')
                    raise Exception(f"RunningHub API错误: {error_msg}")

                data = result.get('data', {})
                task_id = data.get('taskId')
                wss_url = data.get('netWssUrl')

                if not task_id:
                    raise Exception("未返回任务ID")
                # WebSocket URL是可选的，不强制要求

            # 等待任务完成（使用传统轮询方式，避免WebSocket复杂性）
            return await self._wait_for_completion_polling(session, task_id, api_key, timeout, aspect_ratio)

    async def _wait_for_completion_polling(self, session: aiohttp.ClientSession,
                                         task_id: str, api_key: str, timeout: int, aspect_ratio: str) -> Dict:
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
            await asyncio.sleep(3)  # Qwen生成可能需要更长时间

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

                                            # 处理多个图像URL
                                            images = []
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

                                                            # 保持RunningHub生成的原始比例，不强制resize

                                                            # 转换为ComfyUI格式的tensor
                                                            image_np = np.array(pil_image).astype(np.float32) / 255.0
                                                            image_tensor = torch.from_numpy(image_np)
                                                            images.append(image_tensor)

                                            if images:
                                                return {
                                                    'images': images,
                                                    'task_id': task_id,
                                                    'count': len(images)
                                                }
                                            else:
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
                                raise Exception(f"Qwen任务失败: {task_id}")

                            elif task_status in ['RUNNING', 'QUEUED']:
                                # 继续等待
                                continue

            except Exception as e:
                # 如果是最终错误，抛出；否则继续轮询
                if "任务失败" in str(e) or "输出中未找到" in str(e):
                    raise
                continue

        raise Exception(f"Qwen任务超时: {task_id} ({timeout}秒)")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "RunningHubQwenTextToImage": RunningHubQwenTextToImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubQwenTextToImage": "RunningHub Qwen文生图"
}
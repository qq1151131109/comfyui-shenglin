"""
RunningHub InfiniteTalk数字人批量生成节点
基于RunningHub高级API，使用工作流ID 1960943620918579202
支持批量数字人视频生成，从音频文件夹和视频文件夹自动匹配生成
"""

import asyncio
import aiohttp
import json
import time
import os
import ssl
import random
import glob
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import folder_paths

class BatchInfiniteTalkVideo:
    """
    RunningHub InfiniteTalk批量数字人视频生成节点

    遍历音频文件夹中的所有音频，每个音频随机从视频文件夹中选择一个视频进行数字人生成
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 核心输入
                "audio_folder": ("STRING", {
                    "default": "",
                    "tooltip": "音频文件夹路径（通常来自主页批量下载节点的输出）"
                }),
                "video_folder": ("STRING", {
                    "default": "",
                    "tooltip": "人像视频文件夹路径（随机选择视频）"
                }),
                "output_prefix": ("STRING", {
                    "default": "infinitetalk_batch",
                    "tooltip": "输出文件夹前缀"
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
    RETURN_NAMES = ("output_folder", "status")
    FUNCTION = "batch_generate"
    CATEGORY = "🔥 Shenglin/RunningHub"
    DESCRIPTION = "批量生成InfiniteTalk数字人视频"

    def __init__(self):
        self.api_base = "https://www.runninghub.cn"
        self.workflow_id = "1960943620918579202"
        self.upload_timeout = 300  # 5分钟上传超时
        self.task_timeout = 1800   # 30分钟任务超时

        # 支持的音频和视频格式
        self.audio_extensions = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    def get_audio_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有音频文件"""
        audio_files = []
        if not os.path.exists(folder_path):
            return audio_files

        for ext in self.audio_extensions:
            audio_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))

        return sorted(audio_files)

    def get_video_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有视频文件"""
        video_files = []
        if not os.path.exists(folder_path):
            return video_files

        for ext in self.video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))

        return sorted(video_files)

    def batch_generate(
        self, audio_folder, video_folder, output_prefix, api_key, video_width, video_height, fps,
        prompt, steps, cfg_scale, seed, audio_scale, negative_prompt=""
    ):
        """批量生成InfiniteTalk数字人视频"""
        try:
            # 1. 验证参数
            if not api_key:
                return ("", "❌ 错误：请提供RunningHub API密钥")

            if not audio_folder or not os.path.exists(audio_folder):
                return ("", f"❌ 错误：音频文件夹不存在: {audio_folder}")

            if not video_folder or not os.path.exists(video_folder):
                return ("", f"❌ 错误：视频文件夹不存在: {video_folder}")

            # 2. 获取所有音频和视频文件
            audio_files = self.get_audio_files(audio_folder)
            video_files = self.get_video_files(video_folder)

            if not audio_files:
                return ("", f"❌ 错误：音频文件夹中没有找到音频文件: {audio_folder}")

            if not video_files:
                return ("", f"❌ 错误：视频文件夹中没有找到视频文件: {video_folder}")

            print(f"🎬 批量InfiniteTalk数字人视频生成")
            print(f"📁 音频文件夹: {audio_folder}")
            print(f"📁 视频文件夹: {video_folder}")
            print(f"🎵 找到音频文件: {len(audio_files)}个")
            print(f"📹 找到视频文件: {len(video_files)}个")

            # 3. 创建输出目录
            from datetime import datetime
            output_base_folder = folder_paths.get_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 清理输出前缀中的非法字符
            clean_prefix = "".join(c for c in output_prefix if c.isalnum() or c in "._-")
            if not clean_prefix:
                clean_prefix = "infinitetalk_batch"

            output_folder = os.path.join(output_base_folder, f"{clean_prefix}_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)

            # 4. 批量生成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._batch_generate_async(
                    audio_files, video_files, output_folder, api_key,
                    video_width, video_height, fps, prompt, steps, cfg_scale,
                    seed, audio_scale, negative_prompt
                ))
                return result
            finally:
                loop.close()

        except Exception as e:
            print(f"❌ 批量生成异常: {str(e)}")
            return ("", f"❌ 批量生成失败: {str(e)}")

    async def _batch_generate_async(
        self, audio_files, video_files, output_folder, api_key,
        video_width, video_height, fps, prompt, steps, cfg_scale,
        seed, audio_scale, negative_prompt
    ):
        """异步批量生成"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.task_timeout)

        # 统计信息
        total_count = len(audio_files)
        success_count = 0
        fail_count = 0
        result_list = []

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for idx, audio_file in enumerate(audio_files):
                print(f"\n{'='*60}")
                print(f"处理进度: {idx + 1}/{total_count}")
                print(f"音频文件: {os.path.basename(audio_file)}")
                print(f"{'='*60}")

                # 随机选择一个视频
                video_file = random.choice(video_files)
                print(f"📹 随机选择视频: {os.path.basename(video_file)}")

                try:
                    # 生成单个数字人视频
                    result = await self._generate_single_async(
                        session, audio_file, video_file, api_key,
                        video_width, video_height, fps, prompt, steps,
                        cfg_scale, seed, audio_scale, negative_prompt
                    )

                    if result["success"]:
                        success_count += 1
                        video_url = result["video_url"]
                        task_id = result["task_id"]

                        # 下载视频到输出目录
                        audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
                        output_filename = f"{audio_basename}.mp4"
                        output_path = os.path.join(output_folder, output_filename)

                        download_success = await self._download_video(session, video_url, output_path)

                        if download_success:
                            result_list.append({
                                "audio": os.path.basename(audio_file),
                                "video": os.path.basename(video_file),
                                "output": output_filename,
                                "status": "✅ 成功"
                            })
                            print(f"✅ 成功: {output_filename}")
                        else:
                            result_list.append({
                                "audio": os.path.basename(audio_file),
                                "video": os.path.basename(video_file),
                                "output": output_filename,
                                "status": "⚠️ 生成成功但下载失败"
                            })
                            print(f"⚠️ 生成成功但下载失败")
                    else:
                        fail_count += 1
                        result_list.append({
                            "audio": os.path.basename(audio_file),
                            "video": os.path.basename(video_file),
                            "output": "-",
                            "status": f"❌ 失败: {result['error']}"
                        })
                        print(f"❌ 失败: {result['error']}")

                except Exception as e:
                    fail_count += 1
                    result_list.append({
                        "audio": os.path.basename(audio_file),
                        "video": os.path.basename(video_file),
                        "output": "-",
                        "status": f"❌ 异常: {str(e)}"
                    })
                    print(f"❌ 异常: {str(e)}")

        # 生成状态报告
        status_msg = self._generate_status_report(
            total_count, success_count, fail_count,
            output_folder, result_list
        )

        return (output_folder, status_msg)

    async def _generate_single_async(
        self, session, audio_file, video_file, api_key,
        video_width, video_height, fps, prompt, steps,
        cfg_scale, seed, audio_scale, negative_prompt
    ):
        """生成单个数字人视频"""
        try:
            # 1. 上传文件
            video_key = await self.upload_file_async(session, video_file, api_key, "video")
            if not video_key:
                return {"success": False, "error": "视频上传失败"}

            audio_key = await self.upload_file_async(session, audio_file, api_key, "audio")
            if not audio_key:
                return {"success": False, "error": "音频上传失败"}

            # 2. 构建节点参数
            node_info_list = self._build_node_info_list(
                audio_key, video_key, video_width, video_height, fps,
                prompt, negative_prompt, steps, cfg_scale, seed, audio_scale
            )

            # 3. 创建任务
            task_id = await self.create_task_async(session, api_key, node_info_list)
            if not task_id:
                return {"success": False, "error": "创建任务失败"}

            # 4. 等待完成
            video_url = await self.wait_for_completion_async(session, api_key, task_id)
            if not video_url:
                return {"success": False, "error": f"任务执行失败 (ID: {task_id})"}

            return {"success": True, "video_url": video_url, "task_id": task_id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_node_info_list(
        self, audio_key, video_key, video_width, video_height, fps,
        prompt, negative_prompt, steps, cfg_scale, seed, audio_scale
    ):
        """构建节点参数列表"""
        node_info_list = []

        # 音频文件 (节点125)
        node_info_list.append({
            "nodeId": "125",
            "fieldName": "audio",
            "fieldValue": audio_key
        })

        # 视频文件 (节点302)
        node_info_list.append({
            "nodeId": "302",
            "fieldName": "video",
            "fieldValue": video_key
        })

        # 正面提示词 (节点311)
        node_info_list.append({
            "nodeId": "311",
            "fieldName": "text",
            "fieldValue": prompt
        })

        # 负面提示词 (节点241)
        if negative_prompt:
            node_info_list.append({
                "nodeId": "241",
                "fieldName": "negative_prompt",
                "fieldValue": negative_prompt
            })

        # 视频尺寸 (节点304: 高度, 节点305: 宽度)
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

        # 音频不裁切，使用全长
        node_info_list.append({
            "nodeId": "159",
            "fieldName": "start_time",
            "fieldValue": "0:00:00"
        })
        node_info_list.append({
            "nodeId": "159",
            "fieldName": "end_time",
            "fieldValue": "0:10:00"  # 10分钟，实际会根据音频长度自动调整
        })

        # 采样参数 (节点128)
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

        # 音频处理参数 (节点306)
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

        return node_info_list

    async def upload_file_async(self, session: aiohttp.ClientSession, file_path: str, api_key: str, file_type: str) -> Optional[str]:
        """异步上传文件到RunningHub"""
        try:
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

            print(f"  ✓ {file_type}文件上传成功")
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

            async with session.post(
                f"{self.api_base}/task/openapi/create",
                json=task_data,
                headers={"Host": "www.runninghub.cn"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    return None

                result = await response.json()
                if result.get("code") != 0:
                    return None

                task_data = result.get("data", {})
                task_id = str(task_data.get("taskId", ""))

                print(f"  ✓ 任务创建成功: {task_id}")
                return task_id

        except Exception as e:
            print(f"❌ 创建任务异常: {str(e)}")
            return None

    async def wait_for_completion_async(self, session: aiohttp.ClientSession, api_key: str, task_id: str) -> Optional[str]:
        """异步等待任务完成"""
        try:
            start_time = time.time()

            while time.time() - start_time < self.task_timeout:
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
                        await asyncio.sleep(5)
                        continue

                    result = await response.json()
                    if result.get("code") != 0:
                        await asyncio.sleep(5)
                        continue

                    status = result.get("data", "")

                    if status == "SUCCESS":
                        # 获取结果
                        async with session.post(
                            f"{self.api_base}/task/openapi/outputs",
                            json=status_data,
                            headers={"Host": "www.runninghub.cn"},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as result_response:
                            if result_response.status != 200:
                                return None

                            result_data = await result_response.json()
                            if result_data.get("code") != 0:
                                return None

                            outputs = result_data.get("data", [])
                            if outputs and len(outputs) > 0:
                                video_url = outputs[0].get("fileUrl", "")
                                if video_url:
                                    print(f"  ✓ 视频生成完成")
                                    return video_url

                    elif status == "FAILED":
                        return None

                    elif status in ["QUEUED", "RUNNING"]:
                        await asyncio.sleep(10)
                        continue

                    else:
                        await asyncio.sleep(5)

            return None

        except Exception as e:
            print(f"❌ 等待任务异常: {str(e)}")
            return None

    async def _download_video(self, session: aiohttp.ClientSession, video_url: str, output_path: str) -> bool:
        """下载视频文件"""
        try:
            print(f"  ⬇️ 下载视频...")
            async with session.get(video_url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    print(f"  ✓ 视频下载成功: {os.path.basename(output_path)}")
                    return True
                else:
                    print(f"  ❌ 下载失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"  ❌ 下载异常: {str(e)}")
            return False

    def _generate_status_report(self, total, success, fail, output_folder, result_list):
        """生成状态报告"""
        msg = f"{'='*50}\n"
        msg += f"🎉 批量生成完成!\n"
        msg += f"{'='*50}\n\n"
        msg += f"📊 统计信息:\n"
        msg += f"  总计: {total}个\n"
        msg += f"  成功: {success}个\n"
        msg += f"  失败: {fail}个\n"
        if total > 0:
            msg += f"  成功率: {success/total*100:.1f}%\n"
        msg += f"\n📁 输出目录: {output_folder}\n"

        if result_list:
            msg += f"\n📝 详细结果（前20个）:\n"
            for item in result_list[:20]:
                msg += f"  🎵 {item['audio']}\n"
                msg += f"  📹 {item['video']}\n"
                msg += f"  📦 {item['output']}\n"
                msg += f"  {item['status']}\n"
                msg += f"  {'-'*40}\n"

            if len(result_list) > 20:
                msg += f"  ... 还有 {len(result_list) - 20} 个结果\n"

        return msg


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "BatchInfiniteTalkVideo": BatchInfiniteTalkVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchInfiniteTalkVideo": "🎭 批量InfiniteTalk数字人"
}
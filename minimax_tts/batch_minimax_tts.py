"""
批量MiniMax TTS节点
支持整篇文案按换行符分割，批量生成多段语音
"""

import os
import io
import json
import time
import tempfile
import requests
import asyncio
import aiohttp
import ssl
from typing import Dict, Any, Optional, Tuple, List
import torch
import torchaudio
import numpy as np


class BatchMiniMaxTTSNode:
    """批量MiniMax TTS节点 - 支持整篇文案按行处理"""

    def __init__(self):
        self.base_url = "https://api.minimaxi.com/v1/t2a_v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 整篇文案输入
                "script": ("STRING", {
                    "multiline": True,
                    "default": "第一段话，这是开头。\n第二段话，这是发展。\n第三段话，这是结尾。",
                    "tooltip": "整篇文案，每行生成一个音频文件"
                }),
                # API配置
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "MiniMax API密钥"
                }),
                "group_id": ("STRING", {
                    "default": "",
                    "tooltip": "MiniMax Group ID"
                }),
                # 模型选择
                "model": ([
                    "speech-2.5-hd-preview",
                    "speech-2.5-turbo-preview",
                    "speech-02-hd",
                    "speech-02-turbo",
                    "speech-01-hd",
                    "speech-01-turbo"
                ], {"default": "speech-2.5-hd-preview"}),
                # 音色选择
                "voice_id": ([
                    # 基础音色
                    "male-qn-qingse",           # 青涩青年音色
                    "male-qn-jingying",         # 精英青年音色
                    "male-qn-badao",            # 霸道青年音色
                    "male-qn-daxuesheng",       # 青年大学生音色
                    "female-shaonv",            # 少女音色
                    "female-yujie",             # 御姐音色
                    "female-chengshu",          # 成熟女性音色
                    "female-tianmei",           # 甜美女性音色
                    # 主持人音色
                    "presenter_male",           # 男性主持人
                    "presenter_female",         # 女性主持人
                    # 有声书音色
                    "audiobook_male_1",         # 男性有声书1
                    "audiobook_male_2",         # 男性有声书2
                    "audiobook_female_1",       # 女性有声书1
                    "audiobook_female_2",       # 女性有声书2
                ], {"default": "male-qn-qingse"}),
                # 批处理配置
                "max_concurrent": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "最大并发生成数"
                })
            },
            "optional": {
                # 语音参数
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "语速"
                }),
                "volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "音量"
                }),
                "pitch": ("FLOAT", {
                    "default": 0,
                    "min": -10,
                    "max": 10,
                    "step": 1,
                    "tooltip": "音调"
                }),
                "emotion": ([
                    "neutral",
                    "happy",
                    "sad",
                    "angry",
                    "fearful",
                    "disgusted",
                    "surprised"
                ], {"default": "neutral"}),
                # 重试配置
                "retry_count": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "失败重试次数"
                }),
                "retry_delay": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "重试延迟时间（秒）"
                }),
                "timeout": ("INT", {
                    "default": 60,
                    "min": 30,
                    "max": 300,
                    "step": 10,
                    "tooltip": "单个请求超时时间（秒）"
                })
            }
        }

    RETURN_TYPES = ("*", "LIST", "STRING", "STRING")
    RETURN_NAMES = ("audio_list", "file_paths", "durations_info", "processing_log")
    FUNCTION = "generate_batch_tts"
    CATEGORY = "🎵 Shenglin/Audio"
    DESCRIPTION = "批量生成TTS音频，支持整篇文案按行处理"

    def generate_batch_tts(
        self,
        script: str,
        api_key: str,
        group_id: str,
        model: str,
        voice_id: str = "male-qn-qingse",
        max_concurrent: int = 3,
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 0,
        emotion: str = "neutral",
        retry_count: int = 2,
        retry_delay: int = 5,
        timeout: int = 60
    ) -> Tuple[List[Dict[str, Any]], List[str], str, str]:
        """
        批量生成TTS音频

        Returns:
            Tuple: (音频列表, 文件路径列表, 时长信息, 处理日志)
        """

        # 检查必需参数
        if not api_key or not group_id:
            raise ValueError("请提供API Key和Group ID")

        if not script or not script.strip():
            raise ValueError("请输入要转换的文案")

        # 按行分割文案
        lines = [line.strip() for line in script.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("没有找到有效的文案行")

        print(f"📝 开始批量TTS生成 - {len(lines)} 行文案")
        print(f"🎤 使用音色: {voice_id}, 模型: {model}")
        print(f"⚡ 最大并发: {max_concurrent}")

        # 运行异步批量生成
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，使用新线程
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._batch_generate_async(
                        lines, api_key, group_id, model, voice_id, max_concurrent,
                        speed, volume, pitch, emotion, retry_count, retry_delay, timeout
                    ))
                    results = future.result()
            else:
                # 如果循环未运行，直接运行
                results = loop.run_until_complete(
                    self._batch_generate_async(
                        lines, api_key, group_id, model, voice_id, max_concurrent,
                        speed, volume, pitch, emotion, retry_count, retry_delay, timeout
                    )
                )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            results = asyncio.run(
                self._batch_generate_async(
                    lines, api_key, group_id, model, voice_id, max_concurrent,
                    speed, volume, pitch, emotion, retry_count, retry_delay, timeout
                )
            )

        # 处理结果
        audio_waveforms = []
        file_paths = []
        duration_info = []
        processing_log = []

        total_duration_ms = 0
        success_count = 0
        sample_rate = 32000  # 默认采样率

        for i, result in enumerate(results):
            if result is not None and result.get('audio_dict'):
                # 成功的音频
                audio_dict = result['audio_dict']
                waveform = audio_dict['waveform'].squeeze(0)  # 移除batch维度
                audio_waveforms.append(waveform)
                file_paths.append(result['file_path'])
                sample_rate = audio_dict['sample_rate']  # 更新采样率

                duration_ms = result.get('duration_ms', 0)
                duration_info.append(f"段落{i+1}: {duration_ms/1000:.2f}秒")
                total_duration_ms += duration_ms
                success_count += 1

                processing_log.append(f"✅ 段落{i+1}: 成功生成 ({duration_ms/1000:.2f}秒)")
            else:
                # 失败的音频，创建静音占位
                silent_duration = 1.0  # 1秒静音
                silent_samples = int(sample_rate * silent_duration)
                silent_waveform = torch.zeros(1, silent_samples)
                audio_waveforms.append(silent_waveform)
                file_paths.append("")

                duration_info.append(f"段落{i+1}: 生成失败")
                processing_log.append(f"❌ 段落{i+1}: 生成失败")

        # 构建音频列表输出
        audio_list_result = []
        for i, result in enumerate(results):
            if result is not None and result.get('audio_dict'):
                audio_list_result.append(result['audio_dict'])
            else:
                # 失败的音频，创建静音占位
                silent_duration = 1.0  # 1秒静音
                silent_samples = int(sample_rate * silent_duration)
                silent_waveform = torch.zeros(1, silent_samples)
                audio_dict = {
                    "waveform": silent_waveform.unsqueeze(0),
                    "sample_rate": sample_rate
                }
                audio_list_result.append(audio_dict)

        # 生成汇总信息
        durations_summary = f"总时长: {total_duration_ms/1000:.2f}秒, 成功: {success_count}/{len(lines)}\n" + "\n".join(duration_info)
        processing_summary = f"批量TTS处理完成:\n成功: {success_count}/{len(lines)} 段\n" + "\n".join(processing_log)

        print(f"🎵 批量TTS完成: {success_count}/{len(lines)} 段成功")
        print(f"⏱️ 总时长: {total_duration_ms/1000:.2f}秒")

        return (audio_list_result, file_paths, durations_summary, processing_summary)

    async def _batch_generate_async(
        self, lines: List[str], api_key: str, group_id: str, model: str,
        voice_id: str, max_concurrent: int, speed: float, volume: float,
        pitch: float, emotion: str, retry_count: int, retry_delay: int, timeout: int
    ) -> List[Optional[Dict]]:
        """异步批量生成"""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_single(text: str, index: int) -> Optional[Dict]:
            async with semaphore:
                for attempt in range(retry_count + 1):
                    try:
                        if attempt == 0:
                            print(f"🎤 开始生成第 {index+1} 段: {text[:30]}...")
                        else:
                            print(f"🔄 第 {index+1} 段重试 {attempt}/{retry_count}")

                        result = await self._generate_single_tts(
                            text, api_key, group_id, model, voice_id,
                            speed, volume, pitch, emotion, timeout
                        )
                        print(f"✅ 第 {index+1} 段生成成功")
                        return result

                    except Exception as e:
                        error_msg = str(e)
                        print(f"❌ 第 {index+1} 段第 {attempt+1} 次尝试失败: {error_msg}")

                        if attempt < retry_count:
                            print(f"⏳ 等待 {retry_delay} 秒后重试...")
                            await asyncio.sleep(retry_delay)
                        else:
                            print(f"💀 第 {index+1} 段所有重试都失败了")

                return None

        # 并发执行所有任务
        tasks = [generate_single(line, i) for i, line in enumerate(lines)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    async def _generate_single_tts(
        self, text: str, api_key: str, group_id: str, model: str,
        voice_id: str, speed: float, volume: float, pitch: float,
        emotion: str, timeout: int
    ) -> Dict:
        """生成单个TTS音频"""

        url = f"{self.base_url}?GroupId={group_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "text": text.strip(),
            "stream": False,
            "language_boost": "Chinese",
            "output_format": "hex",
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": volume,
                "pitch": int(pitch),
                "emotion": emotion
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            }
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
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败 {response.status}: {error_text}")

                result = await response.json()

                # 检查API响应状态
                if result.get('base_resp', {}).get('status_code', -1) != 0:
                    error_msg = result.get('base_resp', {}).get('status_msg', 'Unknown error')
                    raise Exception(f"MiniMax API错误: {error_msg}")

                # 获取音频数据
                audio_hex = result.get('data', {}).get('audio')
                if not audio_hex:
                    raise Exception("API响应中没有音频数据")

                # 解码hex数据
                audio_bytes = bytes.fromhex(audio_hex)

                # 获取音频信息
                extra_info = result.get('extra_info', {})
                audio_length_ms = extra_info.get('audio_length', 0)

                # 保存临时音频文件
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                temp_file.write(audio_bytes)
                temp_file.close()

                # 加载音频为torch张量
                try:
                    waveform, sample_rate = torchaudio.load(temp_file.name)

                    # 确保是单声道
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # 转换为ComfyUI AUDIO格式
                    audio_dict = {
                        "waveform": waveform.unsqueeze(0),  # 添加batch维度
                        "sample_rate": sample_rate
                    }

                    return {
                        'audio_dict': audio_dict,
                        'file_path': temp_file.name,
                        'duration_ms': audio_length_ms,
                        'text': text
                    }

                except Exception as e:
                    # 如果torchaudio加载失败，清理临时文件
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                    raise Exception(f"音频加载失败: {e}")


# 注册节点
NODE_CLASS_MAPPINGS = {
    "BatchMiniMaxTTS": BatchMiniMaxTTSNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchMiniMaxTTS": "🎤 批量 MiniMax TTS"
}
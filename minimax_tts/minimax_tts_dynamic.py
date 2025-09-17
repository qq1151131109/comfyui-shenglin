"""
MiniMax TTS Node with Dynamic Voice Loading
支持动态获取所有可用音色的TTS节点
"""

import os
import io
import json
import time
import tempfile
import requests
from typing import Dict, Any, Optional, Tuple, List
import torch
import torchaudio
import numpy as np

from .voice_manager import voice_manager


class MiniMaxTTSDynamicNode:
    """MiniMax TTS动态音色节点"""
    
    def __init__(self):
        self.base_url = "https://api.minimaxi.com/v1/t2a_v2"
        self._cached_voices = {}
        self._api_key_cache = ""
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 文本内容
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "你好，这是一个MiniMax TTS测试。"
                }),
                # API配置
                "api_key": ("STRING", {
                    "default": ""
                }),
                "group_id": ("STRING", {
                    "default": ""
                }),
                # 模型选择
                "model": ([
                    "speech-2.5-hd-preview",
                    "speech-2.5-turbo-preview", 
                    "speech-02-hd",
                    "speech-02-turbo",
                    "speech-01-hd",
                    "speech-01-turbo"
                ], {"default": "speech-2.5-hd-preview"})
            },
            "optional": {
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
                    # Beta精品音色
                    "male-qn-qingse-jingpin",   # 青涩青年音色-beta
                    "male-qn-jingying-jingpin", # 精英青年音色-beta
                    "male-qn-badao-jingpin",    # 霸道青年音色-beta
                    "male-qn-daxuesheng-jingpin", # 青年大学生音色-beta
                    "female-shaonv-jingpin",    # 少女音色-beta
                    "female-yujie-jingpin",     # 御姐音色-beta
                    "female-chengshu-jingpin",  # 成熟女性音色-beta
                    "female-tianmei-jingpin",   # 甜美女性音色-beta
                    # 儿童音色
                    "clever_boy",               # 聪明男童
                    "cute_boy",                 # 可爱男童
                    "lovely_girl",              # 萌萌女童
                    "cartoon_pig",              # 卡通猪小琪
                    # 角色音色
                    "bingjiao_didi",            # 病娇弟弟
                    "junlang_nanyou",           # 俊朗男友
                    "chunzhen_xuedi",           # 纯真学弟
                    "lengdan_xiongzhang",       # 冷淡学长
                    "badao_shaoye",             # 霸道少爷
                    "tianxin_xiaoling",         # 甜心小玲
                    "qiaopi_mengmei",           # 俏皮萌妹
                    "wumei_yujie",              # 妩媚御姐
                    "diadia_xuemei",            # 嗲嗲学妹
                    "danya_xuejie",             # 淡雅学姐
                    # 英文音色
                    "Santa_Claus",              # Santa Claus
                    "Grinch",                   # Grinch
                    "Rudolph",                  # Rudolph
                    "Arnold",                   # Arnold
                    "Charming_Santa",           # Charming Santa
                    "Charming_Lady",            # Charming Lady
                    "Sweet_Girl",               # Sweet Girl
                    "Cute_Elf",                 # Cute Elf
                    "Attractive_Girl",          # Attractive Girl
                    "Serene_Woman"              # Serene Woman
                ], {"default": "male-qn-qingse"}),
                # 语音参数
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1
                }),
                "volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "pitch": ("FLOAT", {
                    "default": 0,
                    "min": -10,
                    "max": 10,
                    "step": 1
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
                # 高级设置
                "language_boost": ([
                    "Chinese",
                    "English", 
                    "auto"
                ], {"default": "Chinese"}),
                "refresh_voices": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "LIST")
    RETURN_NAMES = ("audio", "file_path", "image_prompt", "video_prompt", "available_voices")
    FUNCTION = "generate_tts"
    CATEGORY = "🔥 Shenglin/音频处理"
    
    def generate_tts(
        self,
        text: str,
        api_key: str,
        group_id: str,
        model: str,
        voice_id: str = "male-qn-qingse",
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 0,
        emotion: str = "neutral",
        language_boost: str = "Chinese",
        refresh_voices: bool = False
    ) -> Tuple[Dict[str, Any], str, str, str, List[str]]:
        """
        生成TTS音频，并返回分离的提示词
        
        Returns:
            Tuple[Dict[str, Any], str, str, str, List[str]]: (音频数据, 文件路径, 图像提示词, 视频提示词, 可用音色列表)
        """
        
        # 检查必需参数
        if not api_key or not group_id:
            raise ValueError("请提供API Key和Group ID")
        
        if not text or not text.strip():
            raise ValueError("请输入要转换的文本")
        
        try:
            # 获取可用音色列表（如果API key改变或请求刷新）
            available_voices = self._get_available_voices(api_key, refresh_voices)
            
            # 验证选择的音色是否可用
            if voice_id not in available_voices:
                print(f"⚠️  音色 '{voice_id}' 不在可用列表中，使用默认音色")
                voice_id = available_voices[0] if available_voices else "male-qn-qingse"
            
            # 构建API请求
            url = f"{self.base_url}?GroupId={group_id}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "text": text.strip(),
                "stream": False,  # 非流式模式
                "language_boost": language_boost,
                "output_format": "hex",  # 返回hex编码的音频数据
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
            
            print(f"🎵 MiniMax TTS: 生成音频 - 文本长度: {len(text)} 字符")
            print(f"🎤 使用音色: {voice_id}, 模型: {model}")
            print(f"🎭 可用音色总数: {len(available_voices)}")
            
            # 发送API请求
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
            result = response.json()
            
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
            generation_time = time.time() - start_time
            
            # 获取音频信息
            extra_info = result.get('extra_info', {})
            audio_length_ms = extra_info.get('audio_length', 0)
            audio_size = len(audio_bytes)
            
            print(f"✅ TTS生成完成:")
            print(f"   📊 处理时间: {generation_time:.2f}秒")
            print(f"   📏 音频长度: {audio_length_ms/1000:.2f}秒") 
            print(f"   💾 文件大小: {audio_size/1024:.1f}KB")
            
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
                
                print(f"🔊 音频张量形状: {waveform.shape}, 采样率: {sample_rate}Hz")
                
                # 生成分离的提示词
                image_prompt, video_prompt = self._generate_separated_prompts(text)
                
                return (audio_dict, temp_file.name, image_prompt, video_prompt, available_voices)
                
            except Exception as e:
                # 如果torchaudio加载失败，清理临时文件
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                raise Exception(f"音频加载失败: {e}")
        
        except Exception as e:
            print(f"❌ MiniMax TTS生成失败: {e}")
            # 即使生成失败，也返回可用音色列表
            available_voices = self._get_available_voices(api_key, refresh_voices)
            raise
    
    def _generate_separated_prompts(self, text: str) -> Tuple[str, str]:
        """
        将文本转换为图像提示词和视频提示词
        
        Args:
            text: 输入文本
            
        Returns:
            Tuple[str, str]: (图像提示词, 视频提示词)
        """
        # 简单的提示词生成逻辑，可以根据需要优化
        # 图像提示词：描述场景静态画面
        image_prompt = f"A detailed illustration of: {text}. High quality, detailed artwork, cinematic composition, professional lighting."
        
        # 视频提示词：描述动态效果
        video_prompt = f"Camera movement and dynamic effects for: {text}. Smooth camera motion, cinematic transitions, engaging visual flow."
        
        return (image_prompt, video_prompt)
    
    def _get_available_voices(self, api_key: str, force_refresh: bool = False) -> List[str]:
        """获取可用音色列表"""
        
        # 如果API key改变或强制刷新，重新获取
        if (force_refresh or 
            api_key != self._api_key_cache or 
            not self._cached_voices):
            
            print("🔄 获取最新音色列表...")
            
            try:
                # 从MiniMax API获取音色数据
                voice_data = voice_manager.fetch_all_voices(api_key)
                
                # 提取音色ID列表
                voice_list = voice_manager.extract_voice_list(voice_data, include_custom=True)
                
                # 缓存结果
                self._cached_voices = voice_list
                self._api_key_cache = api_key
                
                return voice_list
                
            except Exception as e:
                print(f"❌ 获取音色列表失败: {e}")
                # 返回默认音色列表
                return voice_manager.get_default_voices()
        else:
            # 使用缓存的音色列表
            return self._cached_voices
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 每次都重新生成（不缓存）
        return float("nan")




# 导出节点映射
NODE_CLASS_MAPPINGS = {
    "MiniMaxTTSDynamic": MiniMaxTTSDynamicNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxTTSDynamic": "🎤 MiniMax TTS (Dynamic)"
}
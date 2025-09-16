"""
MiniMax Voice Manager
动态获取和管理MiniMax可用音色
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any
from functools import lru_cache


class MiniMaxVoiceManager:
    """MiniMax音色管理器"""
    
    def __init__(self):
        self.get_voice_url = "https://api.minimaxi.com/v1/get_voice"
        self._voice_cache = {}
        self._cache_timestamp = 0
        self._cache_duration = 3600  # 缓存1小时
        
    @lru_cache(maxsize=1)
    def get_default_voices(self) -> List[str]:
        """获取默认音色列表（作为后备）"""
        return [
            "male-qn-qingse",      # 男声-清澈
            "female-shaonv",       # 女声-少女
            "male-yujie",          # 男声-御姐
            "female-chengshu",     # 女声-成熟
            "male-zhengta",        # 男声-正太
            "female-yujie",        # 女声-御姐
            "male-shaonian",       # 男声-少年
            "female-qingxin",      # 女声-清新
        ]
    
    def fetch_all_voices(self, api_key: str) -> Dict[str, Any]:
        """
        从MiniMax API获取所有可用音色
        
        Args:
            api_key: MiniMax API密钥
            
        Returns:
            Dict包含所有音色信息
        """
        if not api_key:
            print("⚠️  API Key为空，使用默认音色列表")
            return self._get_default_voice_data()
        
        # 检查缓存
        current_time = time.time()
        cache_key = f"voices_{api_key[:8]}"  # 使用API key前8位作为缓存键
        
        if (cache_key in self._voice_cache and 
            current_time - self._cache_timestamp < self._cache_duration):
            print("🔄 使用缓存的音色数据")
            return self._voice_cache[cache_key]
        
        try:
            print("🔍 正在获取MiniMax可用音色...")
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 获取所有类型的音色
            data = {'voice_type': 'all'}
            
            response = requests.post(
                self.get_voice_url, 
                headers=headers, 
                json=data,  # 使用json参数而不是data
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ API请求失败: {response.status_code} - {response.text}")
                return self._get_default_voice_data()
            
            result = response.json()
            
            # 缓存结果
            self._voice_cache[cache_key] = result
            self._cache_timestamp = current_time
            
            # 统计音色数量
            system_count = len(result.get('system_voice', []))
            cloning_count = len(result.get('voice_cloning', []))
            generation_count = len(result.get('voice_generation', []))
            music_count = len(result.get('music_generation', []))
            
            print(f"✅ 音色获取成功:")
            print(f"   📊 系统音色: {system_count}")
            print(f"   🎭 克隆音色: {cloning_count}")  
            print(f"   🤖 生成音色: {generation_count}")
            print(f"   🎵 音乐音色: {music_count}")
            print(f"   📈 总计: {system_count + cloning_count + generation_count + music_count}")
            
            return result
            
        except Exception as e:
            print(f"❌ 获取音色列表失败: {e}")
            return self._get_default_voice_data()
    
    def _get_default_voice_data(self) -> Dict[str, Any]:
        """获取默认音色数据结构"""
        default_voices = self.get_default_voices()
        
        # 模拟系统音色数据结构
        system_voices = []
        for voice_id in default_voices:
            voice_name = self._get_voice_display_name(voice_id)
            system_voices.append({
                "voice_id": voice_id,
                "voice_name": voice_name,
                "description": [voice_name]
            })
        
        return {
            "system_voice": system_voices,
            "voice_cloning": [],
            "voice_generation": [], 
            "music_generation": []
        }
    
    def _get_voice_display_name(self, voice_id: str) -> str:
        """获取音色的显示名称"""
        name_mapping = {
            "male-qn-qingse": "男声-清澈",
            "female-shaonv": "女声-少女",
            "male-yujie": "男声-御姐", 
            "female-chengshu": "女声-成熟",
            "male-zhengta": "男声-正太",
            "female-yujie": "女声-御姐",
            "male-shaonian": "男声-少年",
            "female-qingxin": "女声-清新"
        }
        return name_mapping.get(voice_id, voice_id)
    
    def extract_voice_list(self, voice_data: Dict[str, Any], 
                          include_custom: bool = True) -> List[str]:
        """
        从音色数据中提取voice_id列表
        
        Args:
            voice_data: API返回的音色数据
            include_custom: 是否包含自定义音色(克隆/生成)
            
        Returns:
            音色ID列表
        """
        voice_list = []
        
        # 添加系统音色
        system_voices = voice_data.get('system_voice', [])
        for voice in system_voices:
            voice_id = voice.get('voice_id')
            if voice_id:
                voice_list.append(voice_id)
        
        if include_custom:
            # 添加克隆音色
            cloning_voices = voice_data.get('voice_cloning', [])
            for voice in cloning_voices:
                voice_id = voice.get('voice_id')
                if voice_id:
                    voice_list.append(voice_id)
            
            # 添加生成音色
            generation_voices = voice_data.get('voice_generation', [])
            for voice in generation_voices:
                voice_id = voice.get('voice_id')
                if voice_id:
                    voice_list.append(voice_id)
            
            # 添加音乐音色(人声部分)
            music_voices = voice_data.get('music_generation', [])
            for voice in music_voices:
                voice_id = voice.get('voice_id')
                if voice_id:
                    voice_list.append(voice_id)
        
        # 去重并排序
        voice_list = sorted(list(set(voice_list)))
        
        # 如果列表为空，使用默认列表
        if not voice_list:
            voice_list = self.get_default_voices()
        
        return voice_list
    
    def get_voice_info(self, voice_data: Dict[str, Any], voice_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定音色的详细信息
        
        Args:
            voice_data: API返回的音色数据
            voice_id: 要查询的音色ID
            
        Returns:
            音色详细信息，如果未找到则返回None
        """
        # 在系统音色中查找
        for voice in voice_data.get('system_voice', []):
            if voice.get('voice_id') == voice_id:
                return {
                    'type': 'system',
                    'voice_id': voice_id,
                    'name': voice.get('voice_name', voice_id),
                    'description': voice.get('description', [])
                }
        
        # 在克隆音色中查找
        for voice in voice_data.get('voice_cloning', []):
            if voice.get('voice_id') == voice_id:
                return {
                    'type': 'cloning',
                    'voice_id': voice_id,
                    'description': voice.get('description', []),
                    'created_time': voice.get('created_time', '')
                }
        
        # 在生成音色中查找
        for voice in voice_data.get('voice_generation', []):
            if voice.get('voice_id') == voice_id:
                return {
                    'type': 'generation',
                    'voice_id': voice_id,
                    'description': voice.get('description', []),
                    'created_time': voice.get('created_time', '')
                }
        
        # 在音乐音色中查找
        for voice in voice_data.get('music_generation', []):
            if voice.get('voice_id') == voice_id:
                return {
                    'type': 'music',
                    'voice_id': voice_id,
                    'instrumental_id': voice.get('instrumental_id', ''),
                    'created_time': voice.get('created_time', '')
                }
        
        return None
    
    def format_voice_list_for_ui(self, voice_list: List[str], voice_data: Dict[str, Any]) -> List[str]:
        """
        为UI格式化音色列表，添加描述信息
        
        Args:
            voice_list: 音色ID列表
            voice_data: 音色详细数据
            
        Returns:
            格式化后的音色列表
        """
        formatted_list = []
        
        for voice_id in voice_list:
            voice_info = self.get_voice_info(voice_data, voice_id)
            
            if voice_info:
                voice_type = voice_info.get('type', 'unknown')
                voice_name = voice_info.get('name', voice_id)
                
                if voice_type == 'system':
                    # 系统音色显示名称
                    display_name = f"{voice_id} ({voice_name})"
                else:
                    # 自定义音色显示类型
                    type_label = {
                        'cloning': '克隆', 
                        'generation': '生成',
                        'music': '音乐'
                    }.get(voice_type, voice_type)
                    display_name = f"{voice_id} [{type_label}]"
            else:
                display_name = voice_id
            
            formatted_list.append(display_name)
        
        return formatted_list


# 创建全局音色管理器实例
voice_manager = MiniMaxVoiceManager()
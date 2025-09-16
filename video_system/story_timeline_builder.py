"""
故事时间轴构建节点 - 复刻Coze Node 163547
StoryTimelineBuilder: 微秒级精度的时间轴计算和智能字幕分割
"""

import json
import re
from typing import List, Dict, Any, Tuple

class StoryTimelineBuilder:
    """
    复刻Coze时间轴构建功能
    - 微秒级时间轴精度
    - 智能字幕分割（25字符限制）
    - 多轨道数据整合
    """
    
    def __init__(self):
        self.SUB_CONFIG = {
            "MAX_LINE_LENGTH": 25,  # 与Coze完全一致
            "SPLIT_PRIORITY": ['。','！','？','，',',','：',':','、','；',';',' '],  # 分割优先级
            "TIME_PRECISION": 3  # 微秒精度
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenes_data": ("STRING", {"multiline": True}),  # 场景数据JSON
                "audio_urls": ("LIST",),  # 音频URL列表
                "image_urls": ("LIST",),  # 图像URL列表
                "durations_microseconds": ("LIST",),  # 时长列表（微秒）
                "title": ("STRING", {"default": ""}),  # 2字标题
                "character_image_url": ("STRING", {"default": ""}),  # 主角图像URL
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio_timeline", "video_timeline", "subtitle_timeline", "title_timeline", "animation_timeline")
    
    FUNCTION = "build_timeline"
    CATEGORY = "StoryVideoGenerator"
    
    def build_timeline(self, scenes_data: str, audio_urls: List[str], image_urls: List[str], 
                      durations_microseconds: List[int], title: str, character_image_url: str) -> Tuple[str, str, str, str, str]:
        """构建完整的视频时间轴"""
        try:
            scenes = json.loads(scenes_data)
            
            # 1. 构建音频时间轴
            audio_timeline = self._build_audio_timeline(audio_urls, durations_microseconds)
            
            # 2. 构建视频时间轴（包含动画配置）
            video_timeline = self._build_video_timeline(image_urls, durations_microseconds, character_image_url)
            
            # 3. 构建字幕时间轴（智能分割）
            subtitle_timeline = self._build_subtitle_timeline(scenes, durations_microseconds)
            
            # 4. 构建标题时间轴
            title_timeline = self._build_title_timeline(title, durations_microseconds)
            
            # 5. 构建动画时间轴
            animation_timeline = self._build_animation_timeline(durations_microseconds, len(scenes))
            
            return (
                json.dumps(audio_timeline, ensure_ascii=False),
                json.dumps(video_timeline, ensure_ascii=False), 
                json.dumps(subtitle_timeline, ensure_ascii=False),
                json.dumps(title_timeline, ensure_ascii=False),
                json.dumps(animation_timeline, ensure_ascii=False)
            )
            
        except Exception as e:
            print(f"TimelineBuilder error: {e}")
            return ("{}", "{}", "{}", "{}", "{}")
    
    def _build_audio_timeline(self, audio_urls: List[str], durations: List[int]) -> List[Dict]:
        """构建音频时间轴"""
        audio_timeline = []
        current_time = 0
        
        for i, (audio_url, duration) in enumerate(zip(audio_urls, durations)):
            audio_timeline.append({
                "track": "main_audio",
                "audio_url": audio_url,
                "duration": duration,
                "start": current_time,
                "end": current_time + duration,
                "volume": 1.0
            })
            current_time += duration
        
        # 添加背景音乐（全程）
        total_duration = current_time
        audio_timeline.append({
            "track": "background_music",
            "audio_url": "故事背景音乐.MP3",
            "duration": total_duration,
            "start": 0,
            "end": total_duration,
            "volume": 0.3
        })
        
        # 添加开场音效
        audio_timeline.append({
            "track": "opening_sound",
            "audio_url": "故事开场音效.MP3", 
            "duration": 4884897,  # 4.88秒
            "start": 0,
            "end": 4884897,
            "volume": 0.8
        })
        
        return audio_timeline
    
    def _build_video_timeline(self, image_urls: List[str], durations: List[int], character_image_url: str) -> List[Dict]:
        """构建视频时间轴"""
        video_timeline = []
        current_time = 0
        
        # 添加主角图像（第一段显示）
        if character_image_url and durations:
            video_timeline.append({
                "track": "character",
                "image_url": character_image_url,
                "start": 0,
                "end": durations[0],
                "width": 1440,
                "height": 1080,
                "scale_x": 2.0,  # 主角图像放大2倍
                "scale_y": 2.0,
                "layer": 2  # 上层
            })
        
        # 添加场景图像
        for i, (image_url, duration) in enumerate(zip(image_urls, durations)):
            # 奇偶交替动画配置
            if i % 2 == 0:
                in_animation = "轻微放大"
                in_animation_duration = 100000  # 0.1秒
            else:
                in_animation = None
                in_animation_duration = 0
            
            video_timeline.append({
                "track": "scenes",
                "image_url": image_url,
                "start": current_time,
                "end": current_time + duration,
                "width": 1440,
                "height": 1080,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "layer": 1,  # 背景层
                "in_animation": in_animation,
                "in_animation_duration": in_animation_duration
            })
            current_time += duration
        
        return video_timeline
    
    def _build_subtitle_timeline(self, scenes: List[Dict], durations: List[int]) -> List[Dict]:
        """构建字幕时间轴 - 智能分割算法"""
        subtitle_timeline = []
        current_time = 0
        
        for scene_idx, (scene, total_duration) in enumerate(zip(scenes, durations)):
            caption = scene.get("cap", "")
            if not caption:
                current_time += total_duration
                continue
            
            # 智能分割字幕
            phrases = self._split_long_phrase(caption, self.SUB_CONFIG["MAX_LINE_LENGTH"])
            
            if not phrases:
                current_time += total_duration
                continue
            
            # 按字符数比例分配时间
            total_chars = sum(len(phrase) for phrase in phrases)
            phrase_start_time = current_time
            
            for phrase_idx, phrase in enumerate(phrases):
                if phrase_idx == len(phrases) - 1:  # 最后一个片段
                    phrase_duration = current_time + total_duration - phrase_start_time
                else:
                    char_ratio = len(phrase) / total_chars
                    phrase_duration = int(total_duration * char_ratio)
                
                subtitle_timeline.append({
                    "text": phrase,
                    "start": phrase_start_time,
                    "end": phrase_start_time + phrase_duration,
                    "font_size": 7,  # 与Coze一致
                    "color": "#FFFFFF",
                    "border_color": "#000000",
                    "alignment": "center",
                    "position_y": -810  # 底部位置
                })
                
                phrase_start_time += phrase_duration
            
            current_time += total_duration
        
        return subtitle_timeline
    
    def _split_long_phrase(self, text: str, max_len: int) -> List[str]:
        """智能字幕分割 - 完全复刻Coze算法"""
        if len(text) <= max_len:
            return [text]
        
        # 按优先级查找分隔符
        for delimiter in self.SUB_CONFIG["SPLIT_PRIORITY"]:
            pos = text.rfind(delimiter, 0, max_len)
            if pos > 0:
                split_pos = pos + 1
                first_part = text[:split_pos].strip()
                remaining_part = text[split_pos:].strip()
                return [first_part] + self._split_long_phrase(remaining_part, max_len)
        
        # 汉字边界保护
        for i in range(min(max_len, len(text)) - 1, 0, -1):
            if '\u4e00' <= text[i] <= '\u9fff':  # 汉字范围检查
                first_part = text[:i+1].strip()
                remaining_part = text[i+1:].strip()
                return [first_part] + self._split_long_phrase(remaining_part, max_len)
        
        # 强制分割
        split_pos = min(max_len, len(text))
        first_part = text[:split_pos].strip()
        remaining_part = text[split_pos:].strip()
        
        result = [first_part]
        if remaining_part:
            result.extend(self._split_long_phrase(remaining_part, max_len))
        
        return result
    
    def _build_title_timeline(self, title: str, durations: List[int]) -> List[Dict]:
        """构建标题时间轴"""
        if not title or not durations:
            return []
        
        # 标题只在第一段显示
        return [{
            "text": title,
            "start": 0,
            "end": durations[0],
            "font_size": 40,  # 与Coze一致
            "color": "#000000",  # 黑色文字
            "border_color": "#FFFFFF",  # 白色边框
            "alignment": "center",
            "position_y": 0,  # 中心位置
            "letter_spacing": 26,  # 26像素字间距
            "font_family": "书南体",
            "in_animation": "弹入"  # 入场动画
        }]
    
    def _build_animation_timeline(self, durations: List[int], scene_count: int) -> List[Dict]:
        """构建动画时间轴"""
        animation_timeline = []
        current_time = 0
        
        for i in range(scene_count):
            if i >= len(durations):
                break
                
            duration = durations[i]
            
            if i == 0:  # 主角特殊动画：2.0→1.2→1.0
                animation_timeline.extend([
                    {
                        "target": "character",
                        "property": "scale",
                        "keyframes": [
                            {"time": 0, "value": 2.0, "easing": "linear"},
                            {"time": 533333, "value": 1.2, "easing": "linear"},  # 0.533秒
                            {"time": duration, "value": 1.0, "easing": "linear"}
                        ]
                    }
                ])
            else:  # 场景图奇偶交替：1.0↔1.5
                scene_idx = i - 1
                if scene_idx % 2 == 0:  # 偶数：1.0→1.5
                    start_scale, end_scale = 1.0, 1.5
                else:  # 奇数：1.5→1.0
                    start_scale, end_scale = 1.5, 1.0
                
                animation_timeline.append({
                    "target": f"scene_{i}",
                    "property": "scale",
                    "keyframes": [
                        {"time": current_time, "value": start_scale, "easing": "linear"},
                        {"time": current_time + duration, "value": end_scale, "easing": "linear"}
                    ]
                })
            
            current_time += duration
        
        return animation_timeline


# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "StoryTimelineBuilder": StoryTimelineBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryTimelineBuilder": "Story Timeline Builder",
}
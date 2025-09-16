"""
故事动画处理节点 - 复刻Coze Node 120984/146723
StoryAnimationProcessor: 关键帧动画系统
"""

import json
import torch
from typing import List, Dict, Any, Tuple

class StoryAnimationProcessor:
    """
    复刻Coze关键帧动画功能
    - 奇偶交替缩放动画：1.0↔1.5
    - 主角开场动画：2.0→1.2→1.0
    - 微秒级时间精度
    - 线性缓动
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_timeline": ("STRING", {"multiline": True}),  # 动画时间轴JSON
                "video_timeline": ("STRING", {"multiline": True}),      # 视频时间轴JSON
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_animation_data",)
    
    FUNCTION = "process_animations"
    CATEGORY = "StoryVideoGenerator"
    
    def process_animations(self, animation_timeline: str, video_timeline: str) -> Tuple[str]:
        """处理动画数据，生成最终的关键帧配置"""
        try:
            animation_data = json.loads(animation_timeline)
            video_data = json.loads(video_timeline)
            
            # 处理动画关键帧
            processed_keyframes = self._generate_keyframes(animation_data, video_data)
            
            # 构建最终动画配置
            animation_config = {
                "keyframes": processed_keyframes,
                "global_settings": {
                    "easing_function": "linear",
                    "time_unit": "microseconds",
                    "coordinate_system": "center_origin"
                },
                "tracks": self._organize_animation_tracks(processed_keyframes)
            }
            
            return (json.dumps(animation_config, ensure_ascii=False),)
            
        except Exception as e:
            print(f"AnimationProcessor error: {e}")
            return ("{}",)
    
    def _generate_keyframes(self, animation_data: List[Dict], video_data: List[Dict]) -> List[Dict]:
        """生成关键帧数据 - 完全复刻Coze算法"""
        keyframes = []
        
        # 按轨道处理动画
        character_segments = [v for v in video_data if v.get("track") == "character"]
        scene_segments = [v for v in video_data if v.get("track") == "scenes"]
        
        # 处理主角动画（复刻主角特殊动画：2.0→1.2→1.0）
        for segment in character_segments:
            segment_id = f"character_{segment.get('start', 0)}"
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            duration = end_time - start_time
            
            # 主角三段式缩放动画
            keyframes.extend([
                {
                    "segment_id": segment_id,
                    "property": "UNIFORM_SCALE",
                    "offset": 0,  # 相对于片段开始时间
                    "value": 2.0,
                    "easing": "linear",
                    "absolute_time": start_time
                },
                {
                    "segment_id": segment_id,
                    "property": "UNIFORM_SCALE", 
                    "offset": 533333,  # 0.533秒（微秒）
                    "value": 1.2,
                    "easing": "linear",
                    "absolute_time": start_time + 533333
                },
                {
                    "segment_id": segment_id,
                    "property": "UNIFORM_SCALE",
                    "offset": duration,
                    "value": 1.0,
                    "easing": "linear", 
                    "absolute_time": end_time
                }
            ])
        
        # 处理场景图像动画（奇偶交替1.0↔1.5）
        for i, segment in enumerate(scene_segments):
            segment_id = f"scene_{i}"
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            duration = end_time - start_time
            
            # 计算场景索引（跳过主角）
            scene_index = i
            
            # 奇偶交替缩放方向
            if scene_index % 2 == 0:  # 偶数场景：1.0→1.5
                start_scale = 1.0
                end_scale = 1.5
            else:  # 奇数场景：1.5→1.0
                start_scale = 1.5
                end_scale = 1.0
            
            keyframes.extend([
                {
                    "segment_id": segment_id,
                    "property": "UNIFORM_SCALE",
                    "offset": 0,
                    "value": start_scale,
                    "easing": "linear",
                    "absolute_time": start_time
                },
                {
                    "segment_id": segment_id,
                    "property": "UNIFORM_SCALE",
                    "offset": duration,
                    "value": end_scale,
                    "easing": "linear",
                    "absolute_time": end_time
                }
            ])
        
        return keyframes
    
    def _organize_animation_tracks(self, keyframes: List[Dict]) -> Dict[str, List[Dict]]:
        """按轨道组织动画数据"""
        tracks = {}
        
        for keyframe in keyframes:
            segment_id = keyframe.get("segment_id", "unknown")
            track_name = segment_id.split("_")[0]  # character 或 scene
            
            if track_name not in tracks:
                tracks[track_name] = []
            
            tracks[track_name].append({
                "segment_id": segment_id,
                "property": keyframe.get("property"),
                "keyframes": self._extract_segment_keyframes(keyframes, segment_id)
            })
        
        # 去重
        for track_name in tracks:
            seen_segments = set()
            unique_tracks = []
            for track in tracks[track_name]:
                segment_id = track["segment_id"]
                if segment_id not in seen_segments:
                    seen_segments.add(segment_id)
                    unique_tracks.append(track)
            tracks[track_name] = unique_tracks
        
        return tracks
    
    def _extract_segment_keyframes(self, all_keyframes: List[Dict], target_segment_id: str) -> List[Dict]:
        """提取特定片段的所有关键帧"""
        segment_keyframes = []
        
        for keyframe in all_keyframes:
            if keyframe.get("segment_id") == target_segment_id:
                segment_keyframes.append({
                    "time": keyframe.get("offset", 0),
                    "value": keyframe.get("value", 1.0),
                    "easing": keyframe.get("easing", "linear")
                })
        
        # 按时间排序
        segment_keyframes.sort(key=lambda k: k["time"])
        return segment_keyframes


class StoryAnimationApplier:
    """
    动画应用节点 - 将动画配置应用到视频轨道
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_data": ("STRING", {"multiline": True}),        # 视频数据
                "animation_data": ("STRING", {"multiline": True}),    # 动画配置
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("animated_video_data",)
    
    FUNCTION = "apply_animations"
    CATEGORY = "StoryVideoGenerator"
    
    def apply_animations(self, video_data: str, animation_data: str) -> Tuple[str]:
        """将动画配置应用到视频数据"""
        try:
            video = json.loads(video_data)
            animations = json.loads(animation_data)
            
            # 应用动画到视频轨道
            animated_video = self._merge_animation_with_video(video, animations)
            
            return (json.dumps(animated_video, ensure_ascii=False),)
            
        except Exception as e:
            print(f"AnimationApplier error: {e}")
            return (video_data,)  # 返回原始视频数据
    
    def _merge_animation_with_video(self, video_data: List[Dict], animation_config: Dict) -> List[Dict]:
        """将动画配置合并到视频数据中"""
        animated_video = []
        keyframes = animation_config.get("keyframes", [])
        
        # 为每个视频片段添加动画信息
        for i, video_segment in enumerate(video_data):
            animated_segment = video_segment.copy()
            track = video_segment.get("track", "")
            
            # 查找对应的动画关键帧
            segment_animations = []
            for keyframe in keyframes:
                segment_id = keyframe.get("segment_id", "")
                if (track == "character" and "character" in segment_id) or \
                   (track == "scenes" and "scene" in segment_id and segment_id.endswith(str(i))):
                    segment_animations.append(keyframe)
            
            # 添加动画配置
            if segment_animations:
                animated_segment["animations"] = segment_animations
                animated_segment["has_animation"] = True
            else:
                animated_segment["has_animation"] = False
            
            animated_video.append(animated_segment)
        
        return animated_video


# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "StoryAnimationProcessor": StoryAnimationProcessor,
    "StoryAnimationApplier": StoryAnimationApplier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryAnimationProcessor": "Story Animation Processor",
    "StoryAnimationApplier": "Story Animation Applier",
}
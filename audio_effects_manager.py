#!/usr/bin/env python3
"""
音效管理器 - 管理本地音效库的加载和使用
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class AudioEffectsManager:
    """音效库管理器"""

    def __init__(self, effects_dir: str = None):
        """
        初始化音效管理器

        Args:
            effects_dir: 音效库目录路径，默认为项目下的assets/audio_effects
        """
        if effects_dir is None:
            # 自动确定音效库路径（在ComfyUI节点目录中）
            current_dir = Path(__file__).parent
            effects_dir = current_dir / "assets" / "audio_effects"

        self.effects_dir = Path(effects_dir)
        self.config_file = self.effects_dir / "audio_library_config.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载音效库配置文件"""
        if not self.config_file.exists():
            return self._create_default_config()

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载音效库配置失败: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict:
        """创建默认配置"""
        return {
            "version": "1.0.0",
            "categories": {
                "opening": {"name": "开场音效", "files": []},
                "background": {"name": "背景音乐", "files": []},
                "ambient": {"name": "环境音效", "files": []}
            }
        }

    def get_audio_file(self, category: str, name: str = None) -> Optional[str]:
        """
        获取音效文件路径

        Args:
            category: 音效分类 (opening, background, ambient)
            name: 具体音效名称，不指定则随机选择

        Returns:
            音效文件的绝对路径，如果找不到返回None
        """
        if category not in self.config["categories"]:
            print(f"❌ 音效分类 '{category}' 不存在")
            return None

        files = self.config["categories"][category]["files"]
        if not files:
            print(f"❌ 分类 '{category}' 中没有音效文件")
            return None

        # 选择音效文件
        if name is None:
            # 随机选择
            selected_file = random.choice(files)
        else:
            # 根据名称查找
            selected_file = None
            for file_info in files:
                if file_info.get("name") == name or file_info.get("filename") == name:
                    selected_file = file_info
                    break

            if selected_file is None:
                print(f"❌ 在分类 '{category}' 中找不到音效 '{name}'")
                return None

        # 构建文件路径
        file_path = self.effects_dir / category / selected_file["filename"]

        if not file_path.exists():
            print(f"❌ 音效文件不存在: {file_path}")
            return None

        return str(file_path.absolute())

    def get_opening_sound(self) -> Optional[str]:
        """获取开场音效"""
        return self.get_audio_file("opening")

    def get_background_music(self) -> Optional[str]:
        """获取背景音乐"""
        return self.get_audio_file("background")

    def get_ambient_sound(self, tags: List[str] = None) -> Optional[str]:
        """
        获取环境音效

        Args:
            tags: 标签列表，用于筛选合适的环境音效

        Returns:
            环境音效文件路径
        """
        if tags is None:
            return self.get_audio_file("ambient")

        # 根据标签筛选音效
        files = self.config["categories"]["ambient"]["files"]
        matching_files = []

        for file_info in files:
            file_tags = file_info.get("tags", [])
            if any(tag in file_tags for tag in tags):
                matching_files.append(file_info)

        if not matching_files:
            print(f"❌ 找不到标签为 {tags} 的环境音效")
            return self.get_audio_file("ambient")  # 回退到随机选择

        selected_file = random.choice(matching_files)
        file_path = self.effects_dir / "ambient" / selected_file["filename"]

        return str(file_path.absolute()) if file_path.exists() else None

    def get_audio_info(self, category: str, filename: str) -> Optional[Dict]:
        """
        获取音效文件信息

        Args:
            category: 音效分类
            filename: 文件名

        Returns:
            音效文件信息字典
        """
        if category not in self.config["categories"]:
            return None

        files = self.config["categories"][category]["files"]
        for file_info in files:
            if file_info.get("filename") == filename:
                return file_info

        return None

    def list_categories(self) -> List[str]:
        """列出所有音效分类"""
        return list(self.config["categories"].keys())

    def list_files(self, category: str) -> List[Dict]:
        """
        列出指定分类下的所有音效文件

        Args:
            category: 音效分类

        Returns:
            音效文件信息列表
        """
        if category not in self.config["categories"]:
            return []

        return self.config["categories"][category]["files"]

    def get_story_audio_config(self, story_duration: float) -> Dict:
        """
        获取故事视频的音效配置

        Args:
            story_duration: 故事总时长（秒）

        Returns:
            音效配置字典，包含开场音效和背景音乐的路径和设置
        """
        config = {
            "opening_sound": None,
            "background_music": None,
            "timeline": []
        }

        # 开场音效
        opening_path = self.get_opening_sound()
        if opening_path:
            opening_info = self.get_audio_info("opening", "opening_sound_effect.mp3")
            opening_duration = opening_info.get("duration", 4.885) if opening_info else 4.885

            config["opening_sound"] = {
                "path": opening_path,
                "start": 0,
                "end": opening_duration,
                "volume": 0.8  # 80% 音量
            }

            config["timeline"].append({
                "type": "opening_sound",
                "start": 0,
                "end": opening_duration,
                "volume": 0.8
            })

        # 背景音乐
        bg_path = self.get_background_music()
        if bg_path:
            config["background_music"] = {
                "path": bg_path,
                "start": 0,
                "end": story_duration,
                "volume": 0.3,  # 30% 音量作为背景
                "loop": True    # 循环播放
            }

            config["timeline"].append({
                "type": "background_music",
                "start": 0,
                "end": story_duration,
                "volume": 0.3,
                "loop": True
            })

        return config

    def validate_library(self) -> Tuple[bool, List[str]]:
        """
        验证音效库完整性

        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []

        # 检查配置文件
        if not self.config_file.exists():
            errors.append("配置文件不存在")

        # 检查目录结构
        for category in self.config["categories"]:
            category_dir = self.effects_dir / category
            if not category_dir.exists():
                errors.append(f"分类目录不存在: {category}")
                continue

            # 检查文件
            files = self.config["categories"][category]["files"]
            for file_info in files:
                filename = file_info.get("filename")
                if not filename:
                    errors.append(f"分类 {category} 中有文件缺少filename字段")
                    continue

                file_path = category_dir / filename
                if not file_path.exists():
                    errors.append(f"音效文件不存在: {file_path}")

        return len(errors) == 0, errors


def test_audio_effects_manager():
    """测试音效管理器功能"""
    print("🧪 测试音效管理器...")

    manager = AudioEffectsManager()

    # 验证音效库
    is_valid, errors = manager.validate_library()
    if is_valid:
        print("✅ 音效库验证通过")
    else:
        print("❌ 音效库验证失败:")
        for error in errors:
            print(f"  - {error}")

    # 测试获取音效文件
    opening = manager.get_opening_sound()
    print(f"开场音效: {opening}")

    background = manager.get_background_music()
    print(f"背景音乐: {background}")

    # 测试故事音效配置
    audio_config = manager.get_story_audio_config(60.0)  # 60秒故事
    print(f"故事音效配置: {json.dumps(audio_config, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    test_audio_effects_manager()
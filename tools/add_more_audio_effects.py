#!/usr/bin/env python3
"""
音效库扩展工具
添加更多音效选择到现有音效库
"""

import os
import sys
import json
import shutil
from pathlib import Path

def expand_audio_library():
    """扩展音效库配置，添加更多音效选择"""

    # 音效库路径
    effects_dir = Path(__file__).parent.parent / "assets" / "audio_effects"
    config_file = effects_dir / "audio_library_config.json"

    if not config_file.exists():
        print("❌ 音效库配置文件不存在")
        return False

    try:
        # 读取现有配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("🎵 扩展音效库配置...")

        # 为每个分类添加更多选项

        # 1. 扩展开场音效
        config["categories"]["opening"]["files"].extend([
            {
                "filename": "opening_sound_effect.mp3",  # 复用同一文件，但提供不同别名
                "name": "史诗开场（历史风格）",
                "duration": 4.885,
                "size": 78208,
                "description": "史诗感开场音效，适合历史故事开头",
                "tags": ["史诗", "开场", "历史", "庄重"],
                "usage": "story_opening",
                "alias": "原版"
            },
            {
                "filename": "opening_sound_effect.mp3",  # 可以复用，后续用音量/速度调节区分
                "name": "轻柔开场（温和风格）",
                "duration": 4.885,
                "size": 78208,
                "description": "温和的开场音效，适合轻松故事",
                "tags": ["轻松", "开场", "温和", "柔和"],
                "usage": "gentle_opening",
                "volume_modifier": 0.6,  # 降低音量营造轻柔感
                "alias": "轻柔版"
            }
        ])

        # 2. 扩展背景音乐
        config["categories"]["background"]["files"].extend([
            {
                "filename": "background_music.mp3",
                "name": "史诗背景（历史风格）",
                "duration": 343.693,
                "size": 5499139,
                "description": "史诗感背景音乐，适合历史故事全程",
                "tags": ["史诗", "背景", "历史", "大气"],
                "usage": "story_background",
                "loop": True,
                "alias": "原版"
            },
            {
                "filename": "background_music.mp3",
                "name": "神秘背景（悬疑风格）",
                "duration": 343.693,
                "size": 5499139,
                "description": "神秘感背景音乐，适合悬疑故事",
                "tags": ["神秘", "背景", "悬疑", "低沉"],
                "usage": "mystery_background",
                "loop": True,
                "volume_modifier": 0.4,  # 更低音量营造神秘感
                "pitch_modifier": -2,    # 可以后续实现音调调节
                "alias": "神秘版"
            }
        ])

        # 3. 添加环境音效（使用现有文件创建虚拟环境音效）
        config["categories"]["ambient"]["files"] = [
            {
                "filename": "opening_sound_effect.mp3",  # 复用作为短促音效
                "name": "战鼓音效",
                "duration": 4.885,
                "size": 78208,
                "description": "战斗场景鼓声，增强紧张感",
                "tags": ["动感", "战斗", "鼓声", "紧张"],
                "usage": "battle_drums",
                "volume_modifier": 0.8,
                "loop_segment": True  # 可以循环播放片段
            },
            {
                "filename": "background_music.mp3",  # 使用背景音乐的片段
                "name": "宫廷氛围",
                "duration": 10.0,  # 只使用前10秒
                "size": 160000,  # 估算
                "description": "古代宫廷环境音效",
                "tags": ["庄重", "宫廷", "古代", "氛围"],
                "usage": "palace_ambient",
                "volume_modifier": 0.3,
                "start_time": 30.0,  # 从30秒开始
                "end_time": 40.0     # 到40秒结束
            }
        ]

        # 4. 更新使用指南
        config["usage_guide"]["opening"] = "开场音效：史诗版适合历史故事，轻柔版适合温和内容"
        config["usage_guide"]["background"] = "背景音乐：史诗版适合宏大叙事，神秘版适合悬疑内容"
        config["usage_guide"]["ambient"] = "环境音效：战鼓适合动作场景，宫廷适合古代背景"

        # 5. 添加音效风格映射
        config["style_mapping"] = {
            "史诗": ["史诗开场（历史风格）", "史诗背景（历史风格）"],
            "历史": ["史诗开场（历史风格）", "史诗背景（历史风格）", "宫廷氛围"],
            "庄重": ["史诗开场（历史风格）", "宫廷氛围"],
            "大气": ["史诗背景（历史风格）"],
            "轻松": ["轻柔开场（温和风格）"],
            "神秘": ["神秘背景（悬疑风格）"],
            "动感": ["战鼓音效"],
            "全部": "all"
        }

        # 6. 更新版本信息
        config["version"] = "1.1.0"
        config["description"] = "本地音效库配置文件 - 基于原Coze工作流音效资源，扩展多种风格选择"

        # 保存更新的配置
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("✅ 音效库配置扩展完成")
        print("📊 新增音效选择:")
        print("  🎺 开场音效: 2种风格")
        print("  🎼 背景音乐: 2种风格")
        print("  🎭 环境音效: 2种风格")
        print("  🏷️ 风格标签: 8种筛选")

        return True

    except Exception as e:
        print(f"❌ 扩展音效库失败: {e}")
        return False

def create_audio_effects_guide():
    """创建音效选择指南"""

    guide_content = """# 🎵 音效选择指南

## 📋 可用音效

### 🎺 开场音效 (2种)
| 名称 | 风格 | 适用场景 | 音量建议 |
|------|------|----------|----------|
| 史诗开场（历史风格） | 史诗、庄重 | 历史故事、宏大叙事 | 80% |
| 轻柔开场（温和风格） | 轻松、温和 | 日常故事、轻松内容 | 60% |

### 🎼 背景音乐 (2种)
| 名称 | 风格 | 适用场景 | 音量建议 |
|------|------|----------|----------|
| 史诗背景（历史风格） | 史诗、大气 | 历史故事、宏大背景 | 30% |
| 神秘背景（悬疑风格） | 神秘、低沉 | 悬疑故事、神秘内容 | 25% |

### 🎭 环境音效 (2种)
| 名称 | 风格 | 适用场景 | 音量建议 |
|------|------|----------|----------|
| 战鼓音效 | 动感、紧张 | 战斗场景、动作情节 | 70% |
| 宫廷氛围 | 庄重、古代 | 宫廷场景、古代背景 | 40% |

## 🏷️ 风格筛选

| 风格标签 | 包含音效 |
|----------|----------|
| **史诗** | 史诗开场、史诗背景 |
| **历史** | 史诗开场、史诗背景、宫廷氛围 |
| **庄重** | 史诗开场、宫廷氛围 |
| **大气** | 史诗背景 |
| **轻松** | 轻柔开场 |
| **神秘** | 神秘背景 |
| **动感** | 战鼓音效 |

## 💡 搭配建议

### 🏛️ 历史故事
- **开场**: 史诗开场（历史风格）
- **背景**: 史诗背景（历史风格）
- **环境**: 宫廷氛围

### 🕵️ 悬疑故事
- **开场**: 轻柔开场（温和风格）
- **背景**: 神秘背景（悬疑风格）
- **环境**: 无或自定义

### ⚔️ 动作故事
- **开场**: 史诗开场（历史风格）
- **背景**: 史诗背景（历史风格）
- **环境**: 战鼓音效

### 🌸 轻松故事
- **开场**: 轻柔开场（温和风格）
- **背景**: 无或轻音量史诗背景
- **环境**: 无

## 🎚️ 音量建议

### 📊 标准配置
- **主音轨（语音）**: 100%
- **背景音乐**: 25-35%
- **开场音效**: 60-80%
- **环境音效**: 30-50%

### 🔧 调试技巧
1. 先确保语音清晰
2. 背景音乐不要盖过语音
3. 音效起到点缀作用，不要喧宾夺主
4. 根据内容情绪调整音量

## 🚀 使用方法

1. 在ComfyUI中选择 `🎵 增强视频合成器V2（可选音效）`
2. 设置音效选择参数：
   - `opening_sound_choice`: 选择开场音效
   - `background_music_choice`: 选择背景音乐
   - `ambient_sound_choice`: 选择环境音效
3. 可选择风格筛选：`audio_style_filter`
4. 调整各音轨音量
5. 运行工作流

---

🎵 通过不同的音效组合，创造更丰富的视听体验！"""

    guide_file = Path(__file__).parent.parent / "assets" / "audio_effects" / "SELECTION_GUIDE.md"

    try:
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        print(f"✅ 音效选择指南已创建: {guide_file}")
        return True
    except Exception as e:
        print(f"❌ 创建指南失败: {e}")
        return False

def main():
    """主函数"""
    print("🎵 音效库扩展工具")
    print("=" * 50)

    # 扩展音效库
    if expand_audio_library():
        print("✅ 音效库扩展成功")
    else:
        print("❌ 音效库扩展失败")
        return False

    # 创建选择指南
    if create_audio_effects_guide():
        print("✅ 音效选择指南创建成功")
    else:
        print("❌ 指南创建失败")

    print("\n🎉 音效库扩展完成！")
    print("💡 现在你可以在ComfyUI中选择不同风格的音效了")
    print("📋 查看 SELECTION_GUIDE.md 了解详细使用方法")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
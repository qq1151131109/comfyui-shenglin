#!/usr/bin/env python3
"""
音效库集成测试脚本
测试增强视频合成器的3轨音频功能
"""

import sys
import os
import torch
import torchaudio
import numpy as np
from PIL import Image

# 添加当前路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def create_test_audio(duration=3.0, sample_rate=22050, frequency=440):
    """创建测试音频"""
    t = torch.linspace(0, duration, int(duration * sample_rate))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    return {
        "waveform": waveform.unsqueeze(0),  # 添加batch维度
        "sample_rate": sample_rate
    }

def create_test_image(width=720, height=1280):
    """创建测试图像"""
    # 创建彩色渐变图像
    image = Image.new("RGB", (width, height))
    pixels = []

    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = int(255 * (x + y) / (width + height))
            pixels.append((r, g, b))

    image.putdata(pixels)

    # 转换为tensor格式
    img_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_array).unsqueeze(0)  # 添加batch维度

def test_enhanced_video_composer():
    """测试增强视频合成器"""
    try:
        print("🧪 开始测试增强视频合成器...")

        # 导入增强视频合成器
        from video_system.enhanced_video_composer import EnhancedVideoComposer

        # 创建节点实例
        composer = EnhancedVideoComposer()
        print("✅ 增强视频合成器初始化成功")

        # 创建测试数据
        print("📊 创建测试数据...")

        # 3个测试音频（模拟3个场景）
        audio_list = [
            create_test_audio(duration=2.0, frequency=440),  # A音
            create_test_audio(duration=2.5, frequency=523),  # C音
            create_test_audio(duration=2.2, frequency=659),  # E音
        ]

        # 3个测试图像
        test_images = []
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # 红、绿、蓝

        for i, color in enumerate(colors):
            image = Image.new("RGB", (720, 1280), color)
            img_array = np.array(image).astype(np.float32) / 255.0
            test_images.append(torch.from_numpy(img_array))

        images_batch = torch.stack(test_images)
        print(f"📊 测试数据：{len(audio_list)}个音频，{images_batch.shape[0]}张图像")

        # 测试1：基础3轨音频合成
        print("\n🎵 测试1：3轨音频系统")
        video_path1, info1 = composer.compose_video_with_effects(
            audio_list=audio_list,
            images=images_batch,
            fps=30,
            width=720,
            height=1280,
            enable_audio_effects=True,
            background_music_volume=0.3,
            opening_sound_volume=0.8,
            voice_volume=1.0
        )

        print(f"✅ 测试1完成")
        print(f"📹 视频路径: {video_path1}")
        print(f"📊 详细信息:\n{info1}")

        # 测试2：单轨音频模式（关闭音效）
        print("\n🔇 测试2：单轨音频模式")
        video_path2, info2 = composer.compose_video_with_effects(
            audio_list=audio_list,
            images=images_batch,
            fps=30,
            width=720,
            height=1280,
            enable_audio_effects=False,
            voice_volume=1.0
        )

        print(f"✅ 测试2完成")
        print(f"📹 视频路径: {video_path2}")
        print(f"📊 详细信息:\n{info2}")

        # 测试3：主角图像开场动画 + 3轨音频
        print("\n👤 测试3：主角图像开场动画")

        # 创建主角图像（橙色）
        character_img = Image.new("RGB", (720, 1280), (255, 165, 0))
        char_array = np.array(character_img).astype(np.float32) / 255.0
        character_image = torch.from_numpy(char_array).unsqueeze(0)

        video_path3, info3 = composer.compose_video_with_effects(
            audio_list=audio_list,
            images=images_batch,
            fps=30,
            width=720,
            height=1280,
            enable_audio_effects=True,
            background_music_volume=0.2,
            opening_sound_volume=0.9,
            voice_volume=0.8,
            character_image=character_image,
            enable_character_intro=True,
            char_intro_scale_start=2.0,
            char_intro_scale_mid=1.2,
            char_intro_scale_end=1.0
        )

        print(f"✅ 测试3完成")
        print(f"📹 视频路径: {video_path3}")
        print(f"📊 详细信息:\n{info3}")

        print("\n🎉 所有测试完成！")
        print("📋 测试总结:")
        print(f"  - 测试1 (3轨音频): {video_path1}")
        print(f"  - 测试2 (单轨音频): {video_path2}")
        print(f"  - 测试3 (主角动画): {video_path3}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_effects_manager():
    """测试音效管理器"""
    try:
        print("\n🧪 测试音效管理器...")

        # 导入音效管理器
        sys.path.append("/home/shenglin/Desktop/ComfyUI/reference/story_video_generator")
        from utils.audio_effects_manager import AudioEffectsManager

        manager = AudioEffectsManager()
        print("✅ 音效管理器初始化成功")

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
        print(f"🎺 开场音效: {opening}")

        background = manager.get_background_music()
        print(f"🎼 背景音乐: {background}")

        # 测试故事音效配置
        config = manager.get_story_audio_config(60.0)
        print(f"⚙️ 故事音效配置生成成功")

        return True

    except Exception as e:
        print(f"❌ 音效管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 增强视频合成器 - 音效库集成测试")
    print("=" * 60)

    # 测试1：音效管理器
    manager_ok = test_audio_effects_manager()

    # 测试2：增强视频合成器
    composer_ok = test_enhanced_video_composer()

    # 总结
    print("\n" + "=" * 60)
    if manager_ok and composer_ok:
        print("🎉 所有测试通过！音效库集成成功")
        print("💡 使用说明:")
        print("  1. 在ComfyUI中使用 '🎵 增强视频合成器（3轨音频）' 节点")
        print("  2. 启用 'enable_audio_effects' 参数")
        print("  3. 调整 'background_music_volume' 和 'opening_sound_volume' 音量")
        print("  4. 享受原Coze工作流同款的音效体验！")
    else:
        print("❌ 部分测试失败，请检查配置")

    return manager_ok and composer_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
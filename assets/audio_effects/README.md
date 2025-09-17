# 本地音效库

## 📁 目录结构
```
audio_effects/
├── opening/                    # 开场音效
│   └── opening_sound_effect.mp3
├── background/                 # 背景音乐
│   └── background_music.mp3
├── ambient/                    # 环境音效 (待扩展)
├── audio_library_config.json   # 音效库配置文件
└── README.md                   # 说明文档
```

## 🎵 音效资源

### 开场音效 (opening/)
- **opening_sound_effect.mp3** (4.9秒, 76KB)
  - 史诗感开场音效，适合历史故事开头
  - 标签: 史诗、开场、历史、庄重

### 背景音乐 (background/)
- **background_music.mp3** (5分44秒, 5.2MB)
  - 史诗感背景音乐，适合历史故事全程播放
  - 支持循环播放，建议音量调低作为背景
  - 标签: 史诗、背景、历史、大气

## 🔧 技术规格
- **格式**: MP3
- **采样率**: 44.1kHz
- **声道**: 立体声 (2通道)
- **比特率**: 128kbps
- **编码器**: LAME 3.101

## 💡 使用指南

### 在故事视频生成器中使用
```python
import json
import os

# 加载音效库配置
with open('assets/audio_effects/audio_library_config.json', 'r') as f:
    audio_config = json.load(f)

# 获取开场音效
opening_sound = audio_config['categories']['opening']['files'][0]['filename']
opening_path = f"assets/audio_effects/opening/{opening_sound}"

# 获取背景音乐
bg_music = audio_config['categories']['background']['files'][0]['filename']
bg_path = f"assets/audio_effects/background/{bg_music}"
```

### 音效时长和用途
- **开场音效**: 4.9秒，在故事开始前播放
- **背景音乐**: 343秒，可循环播放贯穿整个故事

## 🎯 集成到工作流

### 3轨音频系统
1. **主音轨**: TTS生成的旁白语音
2. **背景音乐轨**: background_music.mp3 (循环播放，音量50%)
3. **音效轨**: opening_sound_effect.mp3 (开场时播放)

### 时间轴示例
```
时间轴:
0-5秒:    开场音效 + 背景音乐
5秒-结束:  主音轨 + 背景音乐
```

## 📈 扩展建议

### 可添加的音效类型
- **ambient/**: 环境音效
  - 古代战场音效
  - 宫廷环境音效
  - 自然环境音效
- **transition/**: 转场音效
  - 场景切换音效
  - 时间过渡音效
- **emphasis/**: 强调音效
  - 重要事件音效
  - 情绪强调音效

### 音效来源建议
- **免费资源**: Freesound.org, Zapsplat (免费账户)
- **付费资源**: AudioJungle, Pond5
- **AI生成**: Mubert, AIVA

## ⚖️ 版权说明
当前音效文件来源于原Coze工作流，仅用于学习和开发目的。如需商业使用，请确保获得适当的版权许可。

## 🔄 更新维护
- 配置文件: `audio_library_config.json`
- 添加新音效时请更新配置文件
- 保持目录结构的一致性
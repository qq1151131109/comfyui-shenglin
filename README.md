# ComfyUI Shenglin - 盛林的自定义节点集合

一个功能完整的ComfyUI自定义节点集合，包含AI文生图、语音合成、视频制作等完整工具链。

## 🎯 功能模块

### 🎨 RunningHub API 集成 (3个节点)
- **RunningHub Flux文生图** - 基于Flux模型的批量文生图，支持并发处理
- **RunningHub Qwen高级版** - 支持自定义尺寸(720x1280)和高级参数的Qwen文生图
- **RunningHub Qwen文生图** - 快捷版Qwen文生图，支持横屏/竖屏/正方形比例

### 🎵 MiniMax TTS 语音合成 (3个节点)
- **MiniMax批量TTS** - 多行文本批量语音合成，支持并发处理
- **MiniMax TTS (Dynamic)** - 动态音色TTS，支持所有可用音色
- **批量音频预览** - ComfyUI界面内直接预览音频文件

### 🎬 视频合成系统 (5个节点)
- **视频合成器** - Ken Burns效果，智能缩放动画，内存优化流式处理
- **视频预览器** - ComfyUI界面内直接预览视频文件
- **故事时间轴构建器** - 微秒级精度时间轴计算，智能字幕分割
- **故事动画处理器** - 关键帧动画系统，奇偶交替缩放
- **故事视频合成器** - 5轨道音频合成，双字幕系统

## 🚀 特色功能

- **完整工具链**: 从AI文生图 → 语音合成 → 视频制作的完整流程
- **并发处理**: 支持批量并发，提高生产效率
- **内存优化**: 视频处理使用流式算法，避免OOM问题
- **界面预览**: 支持音频、视频在ComfyUI界面内直接预览
- **参数丰富**: 提供专业级的参数控制选项
- **基于Coze**: 视频系统基于Coze工作流完整复刻

## 📦 安装使用

1. 克隆到ComfyUI自定义节点目录:
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/qq1151131109/comfyui-shenglin.git
```

2. 重启ComfyUI即可使用

## 🔑 API配置

### RunningHub API
需要在 [RunningHub](https://www.runninghub.cn) 注册获取API密钥

### MiniMax API
需要在 [MiniMax](https://www.minimaxi.com) 注册获取API密钥和Group ID

## 🎬 视频合成演示

支持的视频效果：
- Ken Burns缩放动画（1.0↔1.5交替）
- 智能场景转换
- 多轨道音频混合
- 双字幕系统（主字幕+标题字幕）
- 内存优化处理（支持长视频生成）

## 📱 输出格式

- **图片**: ComfyUI IMAGE格式，支持批量输出
- **音频**: ComfyUI AUDIO格式，支持列表输出
- **视频**: 标准视频文件，支持预览和导出

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

⭐ **如果觉得有用，请给个星标支持！**

作者: 盛林 | 版本: 1.0.0
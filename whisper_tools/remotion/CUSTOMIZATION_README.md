# 字幕样式自定义指南 🎨

ComfyUI Remotion字幕系统完全支持自定义调整!

## 🎯 可调整项目

- ✅ **字幕位置** (上下位置、容器高度)
- ✅ **字体大小** (响应式缩放比例)
- ✅ **响应式设计** (自动适配所有分辨率)

## 📚 详细指南

### 1. 字幕位置调整

👉 查看 [POSITION_ADJUSTMENT_GUIDE.md](./POSITION_ADJUSTMENT_GUIDE.md)

**快速使用:**
```bash
# 字幕往上移
node adjust-subtitle-position.mjs --bottom 0.25

# 字幕往下移
node adjust-subtitle-position.mjs --bottom 0.10

# 增加容器高度
node adjust-subtitle-position.mjs --height 0.12
```

### 2. 字体大小调整

👉 查看 [SIZE_ADJUSTMENT_GUIDE.md](./SIZE_ADJUSTMENT_GUIDE.md)

**快速使用:**
```bash
# 放大字体
node adjust-subtitle-size.mjs --size 0.14

# 缩小字体
node adjust-subtitle-size.mjs --size 0.08

# 恢复默认
node adjust-subtitle-size.mjs --size 0.11
```

## ⚡ 快速参考

### 默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 字体大小 | 0.11 | 视频最小边长的11% |
| 底部距离 | 0.18 | 距离底部18% |
| 容器高度 | 0.08 | 容器高度8% |

### 常用场景预设

#### TikTok/抖音风格
```bash
node adjust-subtitle-size.mjs --size 0.12
node adjust-subtitle-position.mjs --bottom 0.18 --height 0.08
```

#### YouTube教学视频
```bash
node adjust-subtitle-size.mjs --size 0.10
node adjust-subtitle-position.mjs --bottom 0.15 --height 0.10
```

#### 电影字幕风格
```bash
node adjust-subtitle-size.mjs --size 0.08
node adjust-subtitle-position.mjs --bottom 0.10 --height 0.06
```

#### 广告大字标题
```bash
node adjust-subtitle-size.mjs --size 0.15
node adjust-subtitle-position.mjs --bottom 0.30 --height 0.12
```

## 📊 实际效果预览

### 1080p 竖屏视频 (1080x1920)

| 设置 | 字体大小 | 底部位置 | 容器高度 |
|------|----------|----------|----------|
| 默认 | 119px | 346px | 154px |
| 大字 | 151px | 346px | 192px |
| 小字 | 86px | 346px | 115px |

### 4K 竖屏视频 (2160x3840)

| 设置 | 字体大小 | 底部位置 | 容器高度 |
|------|----------|----------|----------|
| 默认 | 238px | 691px | 307px |
| 大字 | 302px | 691px | 384px |
| 小字 | 173px | 691px | 230px |

## 🎬 使用流程

### 新手推荐流程:

1. **先用默认设置渲染一个测试视频**
   ```bash
   # 在ComfyUI中使用Remotion字幕节点渲染
   ```

2. **查看效果,决定需要调整什么**
   - 字幕太小? → 增大字体
   - 字幕太大? → 减小字体
   - 字幕太靠上? → 减小bottom值
   - 字幕太靠下? → 增大bottom值
   - 文字被截断? → 增大height值

3. **调整参数**
   ```bash
   cd /path/to/remotion
   node adjust-subtitle-size.mjs --size 0.12
   node adjust-subtitle-position.mjs --bottom 0.20
   ```

4. **重新渲染测试**
   - 不需要重启ComfyUI
   - 直接在ComfyUI中重新执行渲染即可

5. **满意后保存设置**
   - 记录你使用的参数值
   - 可以用于以后的视频

## 🔧 高级自定义

### 为不同样式设置不同参数

虽然调整脚本是批量修改所有样式,但你可以手动编辑特定样式:

**示例: 让"霓虹发光"样式使用更大的字体**

```bash
vim src/CaptionedVideo/NeonGlowStyle.tsx
```

找到并修改:
```typescript
// 从
const DESIRED_FONT_SIZE = Math.min(width, height) * 0.11;

// 改为
const DESIRED_FONT_SIZE = Math.min(width, height) * 0.14;
```

### 创建自己的预设组合

创建一个脚本保存你的常用设置:

```bash
# 创建 my-preset.sh
cat > my-preset.sh << 'EOF'
#!/bin/bash
echo "应用自定义字幕预设..."
node adjust-subtitle-size.mjs --size 0.13
node adjust-subtitle-position.mjs --bottom 0.20 --height 0.10
echo "✅ 完成!"
EOF

chmod +x my-preset.sh

# 使用
./my-preset.sh
```

## ⚠️ 重要提示

1. **响应式保留**: 所有调整都保持响应式特性,自动适配任意分辨率

2. **无需重启**: 调整后直接渲染,无需重启ComfyUI

3. **可恢复**: 随时可以恢复到默认值

4. **备份建议**: 在大量修改前,建议备份整个remotion文件夹

5. **测试建议**: 用短视频测试效果,满意后再处理长视频

## 📝 参数范围建议

| 参数 | 最小值 | 推荐最小 | 默认 | 推荐最大 | 最大值 |
|------|--------|----------|------|----------|--------|
| size | 0.05 | 0.08 | 0.11 | 0.17 | 0.25 |
| bottom | 0.05 | 0.10 | 0.18 | 0.35 | 0.50 |
| height | 0.04 | 0.06 | 0.08 | 0.15 | 0.20 |

## 🆘 常见问题

**Q: 调整后字幕消失了?**
A: 检查bottom值是否太大,导致字幕移出画面。恢复默认值重试。

**Q: 字体调整后在不同分辨率下大小不一致?**
A: 这是响应式设计的正常表现。使用的是比例值,会根据视频大小自动缩放。

**Q: 可以只调整某几个样式吗?**
A: 可以,手动编辑对应的Style.tsx文件即可。

**Q: 调整会影响已经渲染的视频吗?**
A: 不会,只影响新渲染的视频。

**Q: 如何知道我当前使用的参数?**
A: 查看任意Style.tsx文件中的参数值:
```bash
grep "DESIRED_FONT_SIZE" src/CaptionedVideo/MinimalStyle.tsx
grep "bottom: height" src/CaptionedVideo/MinimalStyle.tsx
```

## 📞 需要帮助?

- 📖 详细位置调整: [POSITION_ADJUSTMENT_GUIDE.md](./POSITION_ADJUSTMENT_GUIDE.md)
- 📖 详细大小调整: [SIZE_ADJUSTMENT_GUIDE.md](./SIZE_ADJUSTMENT_GUIDE.md)
- 📖 更新日志: [REMOTION_UPDATE_LOG.md](../REMOTION_UPDATE_LOG.md)

## 🎉 快速开始

**第一次使用?试试这个:**

```bash
cd /Users/shenglin/Library/Mobile\ Documents/com~apple~CloudDocs/code/ComfyUI/custom_nodes/comfyui-shenglin/whisper_tools/remotion

# 应用推荐的社交媒体设置
node adjust-subtitle-size.mjs --size 0.12
node adjust-subtitle-position.mjs --bottom 0.18 --height 0.08

# 查看效果
echo "✅ 设置完成!现在可以在ComfyUI中渲染测试视频了。"
```

---

**最后更新**: 2025-10-01
**版本**: 2.0 (响应式设计)

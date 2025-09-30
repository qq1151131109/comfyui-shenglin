# 字幕位置调整指南

## 📍 当前位置设置

所有字幕样式使用响应式设计,位置参数如下:

- **底部距离**: `height * 0.18` (距离底部18%)
- **容器高度**: `height * 0.08` (高度8%)

## 🎯 如何调整位置

### 方法1: 使用调整脚本 (推荐)

我们提供了一个自动化脚本来批量调整所有样式的位置:

```bash
cd /path/to/whisper_tools/remotion
node adjust-subtitle-position.mjs --bottom 0.15 --height 0.10
```

#### 参数说明:

- `--bottom <比例>`: 字幕容器距离底部的比例 (0.0-1.0)
  - 默认: `0.18` (18%)
  - 数值越大,字幕越靠上
  - 数值越小,字幕越靠下

- `--height <比例>`: 字幕容器的高度比例 (0.0-1.0)
  - 默认: `0.08` (8%)
  - 数值越大,容器越高 (可容纳更多文字)

#### 常用场景:

**场景1: 字幕太靠下,想往上移**
```bash
node adjust-subtitle-position.mjs --bottom 0.25
```
效果: 字幕距离底部25%,比默认高7%

**场景2: 字幕太靠上,想往下移**
```bash
node adjust-subtitle-position.mjs --bottom 0.10
```
效果: 字幕距离底部10%,比默认低8%

**场景3: 容器太小,文字被截断**
```bash
node adjust-subtitle-position.mjs --height 0.12
```
效果: 容器高度提升到12%,可容纳更多文字

**场景4: 同时调整位置和高度**
```bash
node adjust-subtitle-position.mjs --bottom 0.20 --height 0.10
```
效果: 字幕距离底部20%,高度10%

### 方法2: 手动修改单个样式

如果只想调整某个特定样式,可以直接编辑该样式文件:

```bash
vim src/CaptionedVideo/MinimalStyle.tsx
```

找到容器定义部分:
```typescript
const container: React.CSSProperties = {
  justifyContent: "center",
  alignItems: "center",
  top: undefined,
  bottom: height * 0.18,  // 👈 修改这里
  height: height * 0.08,  // 👈 修改这里
};
```

修改 `bottom` 和 `height` 的比例值即可。

## 📐 位置参考图

```
┌─────────────────────────┐
│                         │
│                         │ ← top 区域
│                         │
│      视频内容区域        │
│                         │
│                         │
├─────────────────────────┤ ← height * (1 - bottom - height)
│                         │
│      字幕容器区域        │ ← height * height (容器高度8%)
│                         │
├─────────────────────────┤ ← height * bottom (底部边距18%)
│                         │
│      底部安全区         │
│                         │
└─────────────────────────┘
```

## 🎬 不同视频比例的效果

### 9:16 竖屏 (1080x1920)
- 默认 `bottom: 0.18`: 距底部 345px
- 默认 `height: 0.08`: 容器高度 154px

### 16:9 横屏 (1920x1080)
- 默认 `bottom: 0.18`: 距底部 194px
- 默认 `height: 0.08`: 容器高度 86px

### 1:1 方形 (1080x1080)
- 默认 `bottom: 0.18`: 距底部 194px
- 默认 `height: 0.08`: 容器高度 86px

## ⚠️ 注意事项

1. **底部安全区**: 建议 `bottom` 值不小于 `0.10` (10%),避免字幕被设备底部遮挡

2. **容器高度**: 建议 `height` 值在 `0.06-0.12` (6%-12%) 之间,太小会截断文字,太大会占用过多画面

3. **响应式**: 这些比例会自动适配所有分辨率,修改后在所有视频尺寸上都生效

4. **测试**: 修改后建议用不同分辨率的视频测试效果

## 🔄 恢复默认值

如果调整效果不满意,可以恢复到默认值:

```bash
node adjust-subtitle-position.mjs --bottom 0.18 --height 0.08
```

## 💡 实用建议

### 根据内容类型调整:

- **短句子** (1-5个字): `--bottom 0.15 --height 0.06`
- **普通句子** (6-15个字): `--bottom 0.18 --height 0.08` (默认)
- **长句子** (16-30个字): `--bottom 0.20 --height 0.12`

### 根据视频风格调整:

- **电影风格** (字幕在下方黑边): `--bottom 0.05 --height 0.08`
- **社交媒体** (字幕居中偏下): `--bottom 0.18 --height 0.08` (默认)
- **教学视频** (字幕靠上): `--bottom 0.30 --height 0.10`

## 🛠️ 高级用法

### 批量处理不同样式类别:

如果想对不同类型的样式使用不同位置:

1. 极简风格 (靠下):
```bash
node adjust-subtitle-position.mjs --bottom 0.15 --height 0.06
```

2. 动态效果 (居中):
```bash
node adjust-subtitle-position.mjs --bottom 0.25 --height 0.10
```

3. 3D效果 (靠上):
```bash
node adjust-subtitle-position.mjs --bottom 0.30 --height 0.12
```

## 📝 常见问题

**Q: 调整后需要重启ComfyUI吗?**
A: 不需要。下次渲染视频时会自动使用新的位置。

**Q: 可以给不同样式设置不同位置吗?**
A: 可以,需要手动编辑每个样式文件 (方法2)。

**Q: 位置调整会影响响应式设计吗?**
A: 不会,调整的是比例值,依然保持响应式特性。

**Q: 如何知道某个比例值的实际像素?**
A: 实际像素 = 视频高度 × 比例值。例如 1080p 视频的 0.18 = 194px

---

**更新日期**: 2025-10-01
**相关文件**: `adjust-subtitle-position.mjs`

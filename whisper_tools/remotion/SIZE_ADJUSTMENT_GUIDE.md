# 字幕大小调整指南

## 📏 当前字体大小设置

所有字幕样式使用响应式设计,字体大小参数如下:

- **字体大小**: `Math.min(width, height) * 0.11` (视频最小边长的11%)

这意味着:
- 1080x1920 竖屏 → 字体约 119px
- 1920x1080 横屏 → 字体约 119px
- 1080x1080 方形 → 字体约 119px
- 2160x3840 4K竖屏 → 字体约 238px (自动翻倍)

## 🎯 如何调整字体大小

### 方法1: 使用调整脚本 (推荐)

我们提供了一个自动化脚本来批量调整所有样式的字体大小:

```bash
cd /path/to/whisper_tools/remotion
node adjust-subtitle-size.mjs --size 0.12
```

#### 参数说明:

- `--size <比例>`: 字体大小比例 (0.05-0.20 推荐范围)
  - 默认: `0.11` (11%)
  - 数值越大,字体越大
  - 数值越小,字体越小

#### 常用场景:

**场景1: 字体太小,想放大**
```bash
node adjust-subtitle-size.mjs --size 0.14
```
效果: 字体大小提升到14%,约增大27%

**场景2: 字体太大,想缩小**
```bash
node adjust-subtitle-size.mjs --size 0.08
```
效果: 字体大小降低到8%,约缩小27%

**场景3: 使用超大字体 (标题风格)**
```bash
node adjust-subtitle-size.mjs --size 0.17
```
效果: 字体大小提升到17%,约增大55%

**场景4: 使用小字体 (长句子)**
```bash
node adjust-subtitle-size.mjs --size 0.09
```
效果: 字体大小降低到9%,适合显示更多文字

**场景5: 恢复默认值**
```bash
node adjust-subtitle-size.mjs --size 0.11
```

### 方法2: 手动修改单个样式

如果只想调整某个特定样式,可以直接编辑该样式文件:

```bash
vim src/CaptionedVideo/MinimalStyle.tsx
```

找到字体大小定义:
```typescript
// 响应式字体大小
const DESIRED_FONT_SIZE = Math.min(width, height) * 0.11;  // 👈 修改这里
```

修改 `0.11` 为你想要的比例即可。

## 📐 大小参考表

### 不同比例值的效果

| 比例值 | 描述 | 1080p字体 | 4K字体 | 适用场景 |
|--------|------|-----------|---------|----------|
| 0.06 | 迷你字体 | 65px | 130px | 密集信息展示 |
| 0.08 | 小字体 | 86px | 173px | 长句子,多行文本 |
| 0.09 | 较小 | 97px | 194px | 较长句子 |
| 0.11 | 默认 | 119px | 238px | 标准社交媒体 |
| 0.13 | 较大 | 140px | 281px | 强调重点 |
| 0.14 | 大字体 | 151px | 302px | 醒目显示 |
| 0.17 | 超大 | 184px | 367px | 短标题,标语 |
| 0.20 | 巨大 | 216px | 432px | 单字/词展示 |

### 不同分辨率的实际像素大小

以默认 `0.11` 为例:

| 分辨率 | 最小边长 | 字体大小 |
|--------|----------|----------|
| 1080x1920 (竖屏) | 1080 | 119px |
| 1920x1080 (横屏) | 1080 | 119px |
| 1080x1080 (方形) | 1080 | 119px |
| 720x1280 (720p竖屏) | 720 | 79px |
| 2160x3840 (4K竖屏) | 2160 | 238px |
| 3840x2160 (4K横屏) | 2160 | 238px |

## 🎬 不同内容类型的推荐设置

### 根据文字长度:

**短句 (1-5个字)**
```bash
node adjust-subtitle-size.mjs --size 0.15
```
- 字体大: 162px (1080p)
- 醒目,适合标题式短句

**普通句子 (6-15个字)**
```bash
node adjust-subtitle-size.mjs --size 0.11
```
- 字体: 119px (1080p)
- 默认设置,最通用

**长句子 (16-30个字)**
```bash
node adjust-subtitle-size.mjs --size 0.09
```
- 字体: 97px (1080p)
- 更小,容纳更多文字

**超长文本 (30+个字)**
```bash
node adjust-subtitle-size.mjs --size 0.07
```
- 字体: 76px (1080p)
- 紧凑显示

### 根据视频类型:

**TikTok/抖音短视频**
```bash
node adjust-subtitle-size.mjs --size 0.12
```
- 稍大一点,更吸引注意

**YouTube教学视频**
```bash
node adjust-subtitle-size.mjs --size 0.10
```
- 适中,不遮挡太多内容

**电影字幕风格**
```bash
node adjust-subtitle-size.mjs --size 0.08
```
- 较小,低调不抢戏

**广告/宣传片**
```bash
node adjust-subtitle-size.mjs --size 0.15
```
- 大字体,强调信息

## 💡 组合调整

字体大小和位置可以组合调整,获得最佳效果:

### 大字体 + 靠上位置 (标题风格)
```bash
node adjust-subtitle-size.mjs --size 0.15
node adjust-subtitle-position.mjs --bottom 0.30 --height 0.12
```

### 小字体 + 靠下位置 (传统字幕)
```bash
node adjust-subtitle-size.mjs --size 0.08
node adjust-subtitle-position.mjs --bottom 0.10 --height 0.06
```

### 默认平衡设置
```bash
node adjust-subtitle-size.mjs --size 0.11
node adjust-subtitle-position.mjs --bottom 0.18 --height 0.08
```

## ⚠️ 注意事项

1. **可读性**: 字体太小 (<0.07) 可能在手机上看不清

2. **遮挡问题**: 字体太大 (>0.17) 可能遮挡视频内容

3. **长度适配**:
   - 短句子 → 可以用大字体
   - 长句子 → 应该用小字体

4. **设备兼容**:
   - 手机竖屏观看: 建议 0.10-0.13
   - 电脑横屏观看: 建议 0.08-0.11

5. **响应式保留**: 调整的是比例值,在所有分辨率下都保持一致性

## 🔄 快速测试不同大小

创建一个测试脚本,快速预览不同大小效果:

```bash
# 测试小字体
node adjust-subtitle-size.mjs --size 0.08
# 渲染测试视频
npm run build

# 测试大字体
node adjust-subtitle-size.mjs --size 0.14
# 渲染测试视频
npm run build

# 恢复默认
node adjust-subtitle-size.mjs --size 0.11
```

## 🛠️ 高级技巧

### 为不同样式类别设置不同大小

虽然脚本是批量调整,但你可以分批处理:

**1. 极简风格用标准大小**
```bash
node adjust-subtitle-size.mjs --size 0.11
```

**2. 手动编辑动态效果样式,使用更大字体**
编辑 `ElasticZoomStyle.tsx`, `BouncyBallStyle.tsx` 等:
```typescript
const DESIRED_FONT_SIZE = Math.min(width, height) * 0.14;
```

**3. 手动编辑文字密集样式,使用更小字体**
编辑 `TechWireframeStyle.tsx`, `PixelArtStyle.tsx` 等:
```typescript
const DESIRED_FONT_SIZE = Math.min(width, height) * 0.09;
```

## 📝 常见问题

**Q: 调整后需要重启ComfyUI吗?**
A: 不需要。下次渲染视频时会自动使用新的字体大小。

**Q: 可以给不同样式设置不同大小吗?**
A: 可以,需要手动编辑每个样式文件 (方法2)。

**Q: 字体大小会影响响应式设计吗?**
A: 不会,调整的是比例值,依然保持响应式特性。自动适配所有分辨率。

**Q: 字体太大导致文字被截断怎么办?**
A: 同时增加容器高度:
```bash
node adjust-subtitle-position.mjs --height 0.12
```

**Q: 如何知道某个比例值的实际像素?**
A: 运行脚本时会自动显示不同分辨率下的实际像素大小。
或者手动计算: 实际像素 = min(视频宽度, 视频高度) × 比例值

**Q: 推荐的最佳设置是什么?**
A: 取决于内容:
- 社交媒体短视频: `0.11-0.13`
- 长视频/教学: `0.09-0.11`
- 标题/重点: `0.14-0.17`

## 🎨 实用预设

保存这些命令作为常用预设:

```bash
# 预设1: 社交媒体标准
alias subtitle-social="node adjust-subtitle-size.mjs --size 0.12"

# 预设2: 电影字幕
alias subtitle-movie="node adjust-subtitle-size.mjs --size 0.08"

# 预设3: 广告大字
alias subtitle-ad="node adjust-subtitle-size.mjs --size 0.15"

# 预设4: 教学视频
alias subtitle-edu="node adjust-subtitle-size.mjs --size 0.10"
```

---

**更新日期**: 2025-10-01
**相关文件**: `adjust-subtitle-size.mjs`
**配合使用**: `adjust-subtitle-position.mjs`, `POSITION_ADJUSTMENT_GUIDE.md`

#!/usr/bin/env node

/**
 * 调整所有字幕样式的字体大小参数
 *
 * 使用方法:
 * node adjust-subtitle-size.mjs --size 0.12
 *
 * 参数说明:
 * --size: 字体大小比例 (默认 0.11 = 视频最小边长的11%)
 *
 * 示例:
 * - 更大字体: node adjust-subtitle-size.mjs --size 0.14
 * - 更小字体: node adjust-subtitle-size.mjs --size 0.08
 * - 恢复默认: node adjust-subtitle-size.mjs --size 0.11
 *
 * 参考值:
 * - 0.08: 小字体 (适合长句子)
 * - 0.11: 默认大小 (标准社交媒体)
 * - 0.14: 大字体 (更醒目)
 * - 0.17: 超大字体 (短标题)
 */

import { readdirSync, readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// 解析命令行参数
const args = process.argv.slice(2);
let sizeRatio = 0.11; // 默认值

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--size' && args[i + 1]) {
    sizeRatio = parseFloat(args[i + 1]);
  }
}

console.log(`📏 调整字幕字体大小:`);
console.log(`   大小比例: ${(sizeRatio * 100).toFixed(1)}% (视频最小边长)`);
console.log();

const stylesDir = join(__dirname, 'src', 'CaptionedVideo');

// 获取所有样式文件
const files = readdirSync(stylesDir).filter(f => f.endsWith('Style.tsx'));

let updatedCount = 0;
let skippedCount = 0;

files.forEach(file => {
  const filePath = join(stylesDir, file);
  let content = readFileSync(filePath, 'utf-8');

  // 检查是否包含响应式字体大小定义
  if (content.includes('Math.min(width, height)') && content.includes('DESIRED_FONT_SIZE')) {
    // 替换字体大小比例
    // 匹配模式: Math.min(width, height) * 0.11
    content = content.replace(
      /Math\.min\(width,\s*height\)\s*\*\s*[\d.]+/g,
      `Math.min(width, height) * ${sizeRatio}`
    );

    writeFileSync(filePath, content, 'utf-8');
    updatedCount++;
    console.log(`✅ ${file}`);
  } else {
    skippedCount++;
    console.log(`⏭️  ${file} (未找到响应式字体大小)`);
  }
});

console.log();
console.log(`📊 统计:`);
console.log(`   更新: ${updatedCount} 个文件`);
console.log(`   跳过: ${skippedCount} 个文件`);
console.log();

// 计算不同分辨率下的实际像素大小
const resolutions = [
  { name: '9:16 竖屏 (1080x1920)', width: 1080, height: 1920 },
  { name: '16:9 横屏 (1920x1080)', width: 1920, height: 1080 },
  { name: '1:1 方形 (1080x1080)', width: 1080, height: 1080 },
  { name: '4K竖屏 (2160x3840)', width: 2160, height: 3840 },
];

console.log(`📐 不同分辨率下的实际字体大小:`);
resolutions.forEach(res => {
  const minDim = Math.min(res.width, res.height);
  const actualSize = Math.round(minDim * sizeRatio);
  console.log(`   ${res.name}: ${actualSize}px`);
});

console.log();
console.log(`✨ 完成! 字体大小已调整。`);
console.log(`💡 提示: 可以运行 npm run build 测试效果`);

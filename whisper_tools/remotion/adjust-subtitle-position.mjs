#!/usr/bin/env node

/**
 * 调整所有字幕样式的位置参数
 *
 * 使用方法:
 * node adjust-subtitle-position.mjs --bottom 0.15 --height 0.10
 *
 * 参数说明:
 * --bottom: 字幕容器距离底部的比例 (默认 0.18 = 18%)
 * --height: 字幕容器的高度比例 (默认 0.08 = 8%)
 *
 * 示例:
 * - 更靠上: node adjust-subtitle-position.mjs --bottom 0.25
 * - 更靠下: node adjust-subtitle-position.mjs --bottom 0.10
 * - 更高的容器: node adjust-subtitle-position.mjs --height 0.12
 */

import { readdirSync, readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// 解析命令行参数
const args = process.argv.slice(2);
let bottomRatio = 0.18; // 默认值
let heightRatio = 0.08; // 默认值

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--bottom' && args[i + 1]) {
    bottomRatio = parseFloat(args[i + 1]);
  }
  if (args[i] === '--height' && args[i + 1]) {
    heightRatio = parseFloat(args[i + 1]);
  }
}

console.log(`📐 调整字幕位置参数:`);
console.log(`   底部距离: ${(bottomRatio * 100).toFixed(1)}%`);
console.log(`   容器高度: ${(heightRatio * 100).toFixed(1)}%`);
console.log();

const stylesDir = join(__dirname, 'src', 'CaptionedVideo');

// 获取所有样式文件
const files = readdirSync(stylesDir).filter(f => f.endsWith('Style.tsx'));

let updatedCount = 0;
let skippedCount = 0;

files.forEach(file => {
  const filePath = join(stylesDir, file);
  let content = readFileSync(filePath, 'utf-8');

  // 检查是否包含响应式容器定义
  if (content.includes('bottom: height *') && content.includes('height: height *')) {
    // 替换 bottom 值
    content = content.replace(
      /bottom:\s*height\s*\*\s*[\d.]+/g,
      `bottom: height * ${bottomRatio}`
    );

    // 替换 height 值 (容器高度)
    // 需要区分容器的 height 和其他元素的 height
    // 查找容器定义块中的 height
    content = content.replace(
      /(const\s+container[^}]+height:\s*)height\s*\*\s*[\d.]+/g,
      `$1height * ${heightRatio}`
    );

    writeFileSync(filePath, content, 'utf-8');
    updatedCount++;
    console.log(`✅ ${file}`);
  } else {
    skippedCount++;
    console.log(`⏭️  ${file} (未找到响应式容器)`);
  }
});

console.log();
console.log(`📊 统计:`);
console.log(`   更新: ${updatedCount} 个文件`);
console.log(`   跳过: ${skippedCount} 个文件`);
console.log();
console.log(`✨ 完成! 字幕位置已调整。`);
console.log(`💡 提示: 可以运行 npm run build 测试效果`);

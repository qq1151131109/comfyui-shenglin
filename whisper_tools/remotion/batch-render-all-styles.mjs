import { execSync } from "node:child_process";
import {
  existsSync,
  readdirSync,
  lstatSync,
  mkdirSync,
} from "node:fs";
import path from "path";

// 所有可用的字幕样式
const ALL_STYLES = [
  "CaptionedVideo",
  "NeonGlowStyle",
  "RetroWaveStyle",
  "GlitchStyle",
  "MinimalStyle",
  "GlassmorphismStyle",
  "GradientRainbowStyle",
  "CartoonBubbleStyle",
  "MetallicStyle",
  "NeonBorderStyle",
  "PixelArtStyle",
  "ElegantShadowStyle",
  "Embossed3DStyle",
  "FluidGradientStyle",
  "TechWireframeStyle",
  "BouncyBallStyle",
  "SpinSpiralStyle",
  "ElasticZoomStyle",
  "SwayBeatStyle",
  "ShakeImpactStyle",
  "FireFlameStyle",
  "WaterRippleStyle",
  "ElectricLightningStyle",
  "SmokeMistStyle",
  "StarryParticleStyle",
  "QuantumTeleportStyle",
  "HologramStyle",
  "MagicSpellStyle",
  "CyberpunkHackerStyle",
  "CosmicGalaxyStyle",
  "SpaceTimeWarpStyle",
  "DiamondCrystalStyle",
  "AudioWaveStyle",
  "LiquidMetalStyle",
  "RainbowAuroraStyle",
];

const renderVideoWithStyle = async (videoPath, videoName, style, outputDir) => {
  const baseVideoName = videoName.replace(/\.(mp4|webm|mkv|mov)$/, "");
  const outputFileName = `${baseVideoName}_${style}.mp4`;
  const outputPath = path.join(outputDir, outputFileName);

  // 如果输出文件已存在，跳过
  if (existsSync(outputPath)) {
    console.log(`⏭️  跳过已存在的文件: ${outputFileName}`);
    return;
  }

  console.log(`🎬 正在渲染: ${videoName} -> ${style}`);
  console.log(`   输出: ${outputFileName}`);

  try {
    // 使用 remotion render 命令渲染视频，添加高质量参数
    const relativeVideoPath = path.relative(process.cwd(), videoPath);
    execSync(
      `npx remotion render src/index.ts ${style} "${outputPath}" --props='{"src":"${relativeVideoPath}"}' --overwrite --crf=15 --codec=h264 --audio-codec=aac --video-image-format=png --concurrency=1`,
      {
        stdio: ["ignore", "inherit", "inherit"],
        cwd: process.cwd(),
      }
    );
    console.log(`✅ 完成: ${outputFileName}`);
  } catch (error) {
    console.error(`❌ 渲染失败: ${outputFileName}`, error.message);
  }
};

const processVideo = async (videoPath, videoName, outputDir) => {
  console.log(`\n📹 开始处理视频: ${videoName}`);
  console.log(`   包含 ${ALL_STYLES.length} 种字幕样式`);

  for (let i = 0; i < ALL_STYLES.length; i++) {
    const style = ALL_STYLES[i];
    console.log(`\n[${i + 1}/${ALL_STYLES.length}] 处理样式: ${style}`);
    await renderVideoWithStyle(videoPath, videoName, style, outputDir);
  }

  console.log(`\n✨ 视频 ${videoName} 的所有样式处理完成！`);
};

const processDirectory = async (inputDir, outputDir) => {
  console.log(`\n🔍 扫描目录: ${inputDir}`);

  if (!existsSync(inputDir)) {
    console.error(`❌ 输入目录不存在: ${inputDir}`);
    return;
  }

  // 创建输出目录
  if (!existsSync(outputDir)) {
    console.log(`📁 创建输出目录: ${outputDir}`);
    mkdirSync(outputDir, { recursive: true });
  }

  const entries = readdirSync(inputDir).filter((f) => f !== ".DS_Store");
  const videoFiles = entries.filter((entry) => {
    const fullPath = path.join(inputDir, entry);
    const stat = lstatSync(fullPath);
    return (
      stat.isFile() &&
      (entry.endsWith(".mp4") ||
        entry.endsWith(".webm") ||
        entry.endsWith(".mkv") ||
        entry.endsWith(".mov"))
    );
  });

  if (videoFiles.length === 0) {
    console.log(`⚠️  在目录 ${inputDir} 中未找到视频文件`);
    return;
  }

  console.log(`🎯 找到 ${videoFiles.length} 个视频文件:`);
  videoFiles.forEach((file, index) => {
    console.log(`   ${index + 1}. ${file}`);
  });

  console.log(`\n🚀 开始批量处理，总共将生成 ${videoFiles.length * ALL_STYLES.length} 个视频文件...`);

  for (let i = 0; i < videoFiles.length; i++) {
    const videoFile = videoFiles[i];
    const videoPath = path.join(inputDir, videoFile);
    console.log(`\n🔄 [${i + 1}/${videoFiles.length}] 处理视频文件: ${videoFile}`);
    await processVideo(videoPath, videoFile, outputDir);
  }

  console.log(`\n🎉 所有视频处理完成！`);
  console.log(`   输出目录: ${outputDir}`);
  console.log(`   总共生成了 ${videoFiles.length * ALL_STYLES.length} 个视频文件`);
};

// 主程序
const main = async () => {
  const args = process.argv.slice(2);

  // 默认输入和输出目录
  const defaultInputDir = "/Users/shenglin/Library/Mobile Documents/com~apple~CloudDocs/code/remotion-subtitles/public";
  const defaultOutputDir = path.join(process.cwd(), "output", "all-styles");

  const inputDir = args[0] || defaultInputDir;
  const outputDir = args[1] || defaultOutputDir;

  console.log("🎨 批量字幕样式渲染器");
  console.log("=".repeat(50));
  console.log(`📁 输入目录: ${inputDir}`);
  console.log(`📁 输出目录: ${outputDir}`);
  console.log(`🎬 支持格式: .mp4, .webm, .mkv, .mov`);
  console.log(`🎨 字幕样式: ${ALL_STYLES.length} 种`);
  console.log("=".repeat(50));

  await processDirectory(inputDir, outputDir);
};

// 错误处理
process.on('uncaughtException', (error) => {
  console.error('❌ 未捕获的异常:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ 未处理的Promise拒绝:', reason);
  process.exit(1);
});

// 运行主程序
main().catch((error) => {
  console.error("❌ 程序运行出错:", error);
  process.exit(1);
});
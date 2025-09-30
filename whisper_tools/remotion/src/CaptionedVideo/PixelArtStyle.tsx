// 像素艺术样式 - 8位游戏风格
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TheBoldFont } from "../load-font";
import { fitText } from "@remotion/layout-utils";
import { makeTransform, scale, translateY } from "@remotion/animation-utils";
import { TikTokPage } from "@remotion/captions";

const fontFamily = TheBoldFont;




const PIXEL_COLORS = ["#FF6B35", "#00C9A7", "#4ECDC4", "#FFE66D", "#FF6B9D", "#A8E6CF"];

export const PixelArtStyle: React.FC<{
  readonly enterProgress: number;
  readonly page: TikTokPage;
}> = ({ enterProgress, page }) => {
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  const timeInMs = (frame / fps) * 1000;

  // 响应式字体大小
  const DESIRED_FONT_SIZE = Math.min(width, height) * 0.11;

  // 响应式容器
  const container: React.CSSProperties = {
    justifyContent: "center",
    alignItems: "center",
    top: undefined,
    bottom: height * 0.18,
    height: height * 0.08,
  };

  const fittedText = fitText({
    fontFamily,
    text: page.text,
    withinWidth: width * 0.9,
    textTransform: "uppercase",
  });

  const fontSize = Math.min(DESIRED_FONT_SIZE, fittedText.fontSize);

  return (
    <AbsoluteFill style={container}>
      {/* 像素网格背景 */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)
          `,
          backgroundSize: "8px 8px",
          opacity: 0.3,
        }}
      />
      
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.8, 1])),
            translateY(interpolate(enterProgress, [0, 1], [25, 0])),
          ]),
          // 像素化滤镜
          imageRendering: "pixelated",
          filter: "contrast(1.2) saturate(1.3)",
        }}
      >
        <span>
          {page.tokens.map((t, index) => {
            const startRelativeToSequence = t.fromMs - page.startMs;
            const endRelativeToSequence = t.toMs - page.startMs;

            const isCurrentlyReading = 
              startRelativeToSequence <= timeInMs &&
              endRelativeToSequence > timeInMs;
            
            const hasBeenRead = endRelativeToSequence <= timeInMs;
            const notYetRead = startRelativeToSequence > timeInMs;

            const pixelColor = PIXEL_COLORS[index % PIXEL_COLORS.length];
            
            // 像素闪烁效果
            const shouldFlicker = isCurrentlyReading && frame % 6 < 3;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : (shouldFlicker ? 0.7 : 1),
                  color: isCurrentlyReading 
                    ? "white"
                    : hasBeenRead 
                      ? pixelColor
                      : "white",
                  
                  // 像素块背景
                  background: isCurrentlyReading 
                    ? `linear-gradient(90deg, 
                        ${pixelColor} 0%, 
                        ${pixelColor}E6 25%, 
                        ${pixelColor}CC 50%, 
                        ${pixelColor}E6 75%, 
                        ${pixelColor} 100%)`
                    : "transparent",
                  borderRadius: "0px", // 保持方形像素风格
                  padding: isCurrentlyReading ? "6px 12px" : "0px",
                  margin: isCurrentlyReading ? "0 3px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  // 像素边框
                  border: isCurrentlyReading 
                    ? `3px solid ${shouldFlicker ? '#FFFFFF' : pixelColor}` 
                    : "none",
                  
                  // 像素化变换
                  transform: isCurrentlyReading 
                    ? `scale(${1.15 + (shouldFlicker ? 0.05 : 0)}) translateY(-6px)`
                    : notYetRead 
                      ? "scale(0.85) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  transition: shouldFlicker ? "none" : "all 0.2s steps(4, end)",
                  
                  // 像素阴影
                  boxShadow: isCurrentlyReading 
                    ? `4px 4px 0px ${pixelColor}80,
                       8px 8px 0px ${pixelColor}60,
                       12px 12px 0px ${pixelColor}40,
                       ${shouldFlicker ? '0 0 20px #FFFFFF80' : 'none'}`
                    : hasBeenRead
                      ? `2px 2px 0px ${pixelColor}60`
                      : "2px 2px 0px rgba(0, 0, 0, 0.5)",
                  
                  // 像素文字阴影
                  textShadow: isCurrentlyReading 
                    ? `2px 2px 0px ${pixelColor},
                       4px 4px 0px rgba(0, 0, 0, 0.7),
                       ${shouldFlicker ? '0 0 8px #FFFFFF' : ''}`
                    : hasBeenRead
                      ? `1px 1px 0px ${pixelColor}80`
                      : "2px 2px 0px rgba(0, 0, 0, 0.8)",
                  
                  // 像素字体效果
                  fontWeight: "900",
                  letterSpacing: "1px",
                  
                  // 像素扫描线效果
                  '::before': isCurrentlyReading && !shouldFlicker ? {
                    content: '""',
                    position: "absolute",
                    top: "0",
                    left: "0",
                    right: "0",
                    bottom: "0",
                    backgroundImage: `repeating-linear-gradient(
                      0deg,
                      transparent,
                      transparent 2px,
                      rgba(0,0,0,0.2) 2px,
                      rgba(0,0,0,0.2) 4px
                    )`,
                    pointerEvents: "none",
                  } : undefined,
                  
                  // 额外的像素装饰
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-6px",
                    left: "-6px",
                    width: `${width * 0.006}px`,
                    height: `${height * 0.003}px`,
                    backgroundColor: shouldFlicker ? "#FFFFFF" : pixelColor,
                    animation: `pixel-blink 0.5s steps(2, end) infinite`,
                  } : undefined,
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      <style>{`
        @keyframes pixel-blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </AbsoluteFill>
  );
};
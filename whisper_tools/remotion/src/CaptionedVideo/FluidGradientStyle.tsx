// 流体渐变样式 - 动态流动渐变效果
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

const container: React.CSSProperties = {
  justifyContent: "center",
  alignItems: "center",
  top: undefined,
  bottom: 350,
  height: 150,
};

const DESIRED_FONT_SIZE = 120;

export const FluidGradientStyle: React.FC<{
  readonly enterProgress: number;
  readonly page: TikTokPage;
}> = ({ enterProgress, page }) => {
  const frame = useCurrentFrame();
  const { width, fps } = useVideoConfig();
  const timeInMs = (frame / fps) * 1000;

  const fittedText = fitText({
    fontFamily,
    text: page.text,
    withinWidth: width * 0.9,
    textTransform: "uppercase",
  });

  const fontSize = Math.min(DESIRED_FONT_SIZE, fittedText.fontSize);

  // 动态颜色变化
  const time = frame * 0.02;
  const gradientOffsets = [
    Math.sin(time) * 50 + 50,
    Math.cos(time * 1.2) * 50 + 50,
    Math.sin(time * 0.8) * 50 + 50,
  ];

  return (
    <AbsoluteFill style={container}>
      {/* 动态流体背景 */}
      <div
        style={{
          position: "absolute",
          width: "200%",
          height: "200%",
          background: `
            radial-gradient(circle at ${gradientOffsets[0]}% ${gradientOffsets[1]}%, 
              rgba(139, 69, 19, 0.3) 0%, transparent 50%),
            radial-gradient(circle at ${gradientOffsets[2]}% ${gradientOffsets[0]}%, 
              rgba(255, 20, 147, 0.2) 0%, transparent 50%),
            radial-gradient(circle at ${gradientOffsets[1]}% ${gradientOffsets[2]}%, 
              rgba(30, 144, 255, 0.2) 0%, transparent 50%)
          `,
          filter: "blur(60px)",
          animation: "fluid-move 20s ease-in-out infinite",
        }}
      />
      
      <div
        style={{
          fontSize,
          color: "transparent",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.95, 1])),
            translateY(interpolate(enterProgress, [0, 1], [15, 0])),
          ]),
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

            // 为每个词生成独特的流体渐变
            const wordTime = time + index * 0.5;
            const fluidGradient = `
              linear-gradient(${45 + Math.sin(wordTime) * 90}deg,
                hsl(${200 + Math.sin(wordTime * 0.7) * 60}, 70%, 60%) 0%,
                hsl(${280 + Math.cos(wordTime * 0.5) * 80}, 80%, 70%) 25%,
                hsl(${40 + Math.sin(wordTime * 1.2) * 40}, 90%, 65%) 50%,
                hsl(${320 + Math.cos(wordTime * 0.8) * 60}, 75%, 60%) 75%,
                hsl(${160 + Math.sin(wordTime * 1.5) * 50}, 85%, 55%) 100%)
            `;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 流体渐变文字
                  background: isCurrentlyReading 
                    ? fluidGradient
                    : hasBeenRead
                      ? `linear-gradient(45deg,
                          hsl(${200 + index * 30}, 60%, 70%) 0%,
                          hsl(${280 + index * 30}, 70%, 60%) 100%)`
                      : `linear-gradient(45deg,
                          rgba(255, 255, 255, 0.9) 0%,
                          rgba(255, 255, 255, 0.7) 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundSize: isCurrentlyReading ? "400% 400%" : "200% 200%",
                  
                  // 流体背景容器
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-8px",
                    left: "-12px",
                    right: "-12px",
                    bottom: "-8px",
                    background: `${fluidGradient}`,
                    borderRadius: "20px",
                    filter: "blur(8px)",
                    opacity: 0.3,
                    zIndex: -1,
                    animation: "fluid-background 4s ease-in-out infinite",
                  } : undefined,
                  
                  // 流动的边框
                  border: isCurrentlyReading 
                    ? `2px solid transparent` 
                    : "none",
                  borderRadius: isCurrentlyReading ? "16px" : "0px",
                  backgroundClip: isCurrentlyReading ? "padding-box" : "initial",
                  padding: isCurrentlyReading ? "6px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  transition: "all 0.5s cubic-bezier(0.23, 1, 0.32, 1)",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.08) translateY(-6px)`
                    : notYetRead 
                      ? "scale(0.92) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 流体光晕
                  filter: isCurrentlyReading 
                    ? `drop-shadow(0 0 20px hsl(${200 + Math.sin(wordTime) * 60}, 70%, 60%))`
                    : hasBeenRead
                      ? `drop-shadow(0 0 8px hsl(${200 + index * 30}, 60%, 70%))`
                      : "drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1))",
                  
                  // 动态背景动画
                  animation: isCurrentlyReading 
                    ? `fluid-text 6s ease-in-out infinite` 
                    : "none",
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      <style>{`
        @keyframes fluid-move {
          0%, 100% { transform: translate(-50%, -50%) rotate(0deg) scale(1); }
          25% { transform: translate(-30%, -70%) rotate(90deg) scale(1.1); }
          50% { transform: translate(-70%, -30%) rotate(180deg) scale(0.9); }
          75% { transform: translate(-40%, -60%) rotate(270deg) scale(1.05); }
        }
        
        @keyframes fluid-background {
          0%, 100% { background-size: 400% 400%; background-position: 0% 50%; }
          50% { background-size: 600% 600%; background-position: 100% 50%; }
        }
        
        @keyframes fluid-text {
          0%, 100% { 
            background-size: 400% 400%; 
            background-position: 0% 50%;
          }
          25% { 
            background-size: 600% 600%; 
            background-position: 100% 0%;
          }
          50% { 
            background-size: 400% 400%; 
            background-position: 50% 100%;
          }
          75% { 
            background-size: 600% 600%; 
            background-position: 0% 50%;
          }
        }
      `}</style>
    </AbsoluteFill>
  );
};
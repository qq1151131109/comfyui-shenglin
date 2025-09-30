// 渐变彩虹样式 - 动态彩虹渐变效果
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





export const GradientRainbowStyle: React.FC<{
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

  // 动态彩虹色相偏移
  const hueOffset = (frame * 2) % 360;

  return (
    <AbsoluteFill style={container}>
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [25, 0])),
          ]),
          WebkitTextStroke: "3px black",
          paintOrder: "stroke",
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

            // 为每个词分配不同的彩虹色相
            const wordHue = (hueOffset + index * 60) % 360;
            const rainbowGradient = `linear-gradient(45deg, 
              hsl(${wordHue}, 100%, 70%) 0%, 
              hsl(${(wordHue + 60) % 360}, 100%, 60%) 25%, 
              hsl(${(wordHue + 120) % 360}, 100%, 70%) 50%, 
              hsl(${(wordHue + 180) % 360}, 100%, 60%) 75%, 
              hsl(${(wordHue + 240) % 360}, 100%, 70%) 100%)`;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 彩虹渐变文字
                  background: isCurrentlyReading 
                    ? rainbowGradient
                    : hasBeenRead
                      ? `hsl(${wordHue}, 80%, 75%)`
                      : "white",
                  WebkitBackgroundClip: isCurrentlyReading || hasBeenRead ? "text" : "initial",
                  WebkitTextFillColor: isCurrentlyReading || hasBeenRead ? "transparent" : "white",
                  backgroundSize: isCurrentlyReading ? "300% 300%" : "100% 100%",
                  
                  // 动态背景动画
                  animation: isCurrentlyReading ? "rainbow-shift 2s ease-in-out infinite" : "none",
                  
                  // 彩虹背景框
                  backgroundColor: isCurrentlyReading 
                    ? "rgba(255, 255, 255, 0.15)" 
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "15px" : "0px",
                  padding: isCurrentlyReading ? "6px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  // 彩虹边框
                  border: isCurrentlyReading 
                    ? `2px solid hsl(${wordHue}, 100%, 60%)` 
                    : "none",
                  
                  transition: "all 0.4s ease-out",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.15) translateY(-5px) rotateZ(${Math.sin(frame * 0.1) * 2}deg)`
                    : notYetRead 
                      ? "scale(0.85) translateY(10px)"
                      : "scale(1) translateY(0px)",
                  
                  // 彩虹光晕
                  boxShadow: isCurrentlyReading 
                    ? `0 0 30px hsl(${wordHue}, 100%, 60%), 
                       0 0 60px hsl(${(wordHue + 120) % 360}, 100%, 60%), 
                       0 0 90px hsl(${(wordHue + 240) % 360}, 100%, 60%)`
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 0 10px hsl(${wordHue}, 100%, 50%),
                       2px 2px 8px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? `1px 1px 6px hsl(${wordHue}, 60%, 40%)`
                      : "2px 2px 8px rgba(0, 0, 0, 0.7)",
                  
                  // 滤镜增强
                  filter: isCurrentlyReading 
                    ? `saturate(1.5) brightness(1.2) hue-rotate(${Math.sin(frame * 0.05) * 30}deg)`
                    : hasBeenRead
                      ? "saturate(1.2)"
                      : "none",
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 添加CSS动画 */}
      <style>{`
        @keyframes rainbow-shift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
      `}</style>
    </AbsoluteFill>
  );
};
// 优雅阴影样式 - 精致柔和的阴影效果
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
const ELEGANT_COLORS = ["#E8B4C2", "#B8E6B8", "#E8D4B4", "#B4D4E8", "#D4B4E8", "#B4E8D4"];

export const ElegantShadowStyle: React.FC<{
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

  return (
    <AbsoluteFill style={container}>
      {/* 柔和背景光晕 */}
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            width: `${100 + i * 50}px`,
            height: `${40 + i * 15}px`,
            background: `radial-gradient(ellipse, ${ELEGANT_COLORS[i % ELEGANT_COLORS.length]}20, transparent)`,
            borderRadius: "50%",
            left: `${20 + i * 20}%`,
            top: `${30 + Math.sin(frame * 0.01 + i) * 10}%`,
            filter: "blur(20px)",
            opacity: 0.6,
          }}
        />
      ))}
      
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.95, 1])),
            translateY(interpolate(enterProgress, [0, 1], [18, 0])),
          ]),
          filter: "drop-shadow(0 4px 12px rgba(0, 0, 0, 0.15))",
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

            const elegantColor = ELEGANT_COLORS[index % ELEGANT_COLORS.length];
            
            // 柔和呼吸动画
            const breathe = Math.sin(frame * 0.06) * 0.03 + 1;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? elegantColor
                    : hasBeenRead 
                      ? "rgba(255, 255, 255, 0.85)"
                      : "white",
                  
                  // 优雅背景
                  background: isCurrentlyReading 
                    ? `linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.95) 0%,
                        rgba(248, 250, 252, 0.9) 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "30px" : "0px",
                  padding: isCurrentlyReading ? "10px 22px" : "0px",
                  margin: isCurrentlyReading ? "0 6px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-5px" : "0px",
                  
                  // 柔和边框
                  border: isCurrentlyReading 
                    ? `2px solid ${elegantColor}60` 
                    : "none",
                  
                  transition: "all 0.6s cubic-bezier(0.23, 1, 0.32, 1)",
                  
                  transform: isCurrentlyReading 
                    ? `scale(${1.05 * breathe}) translateY(-8px)`
                    : notYetRead 
                      ? "scale(0.95) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 多层优雅阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 2px 8px ${elegantColor}40,
                       0 8px 24px ${elegantColor}20,
                       0 16px 48px rgba(0, 0, 0, 0.08),
                       inset 0 1px 0 rgba(255, 255, 255, 0.8),
                       inset 0 -1px 0 rgba(0, 0, 0, 0.05)`
                    : hasBeenRead
                      ? `0 2px 8px ${elegantColor}30`
                      : "0 2px 6px rgba(0, 0, 0, 0.1)",
                  
                  // 优雅文字阴影
                  textShadow: isCurrentlyReading 
                    ? "none" // 白色背景上不需要阴影
                    : hasBeenRead
                      ? `0 2px 12px ${elegantColor}50,
                         0 1px 3px rgba(0, 0, 0, 0.2)`
                      : "0 2px 8px rgba(0, 0, 0, 0.3)",
                  
                  // 字体权重
                  fontWeight: isCurrentlyReading ? "800" : "700",
                  
                  // 柔和装饰
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "20%",
                    left: "15%",
                    width: "20%",
                    height: "20%",
                    background: `radial-gradient(circle, rgba(255, 255, 255, 0.8) 0%, transparent 70%)`,
                    borderRadius: "50%",
                    filter: "blur(3px)",
                  } : undefined,
                  
                  // 底部装饰线
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: "-8px",
                    left: "20%",
                    right: "20%",
                    height: "2px",
                    background: `linear-gradient(90deg, transparent, ${elegantColor}60, transparent)`,
                    borderRadius: "1px",
                  } : undefined,
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 装饰性粒子 */}
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={`particle-${i}`}
          style={{
            position: "absolute",
            width: "3px",
            height: "3px",
            background: ELEGANT_COLORS[i % ELEGANT_COLORS.length],
            borderRadius: "50%",
            left: `${15 + i * 15}%`,
            top: `${40 + Math.sin(frame * 0.02 + i) * 15}%`,
            opacity: Math.sin(frame * 0.03 + i * 0.5) * 0.3 + 0.4,
            filter: "blur(1px)",
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
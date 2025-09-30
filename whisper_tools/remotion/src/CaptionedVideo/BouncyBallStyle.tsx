// 弹跳球样式 - 活泼有趣的弹跳动画
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




const BOUNCE_COLORS = ["#FF4757", "#2ED573", "#3742FA", "#FF6348", "#A4B0BE", "#FFA502"];

export const BouncyBallStyle: React.FC<{
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
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.8, 1])),
            translateY(interpolate(enterProgress, [0, 1], [40, 0])),
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

            const bounceColor = BOUNCE_COLORS[index % BOUNCE_COLORS.length];
            
            // 弹跳动画计算
            const bounceFreq = 0.4;
            const bounceHeight = isCurrentlyReading ? 15 : 0;
            const bounceY = Math.abs(Math.sin(frame * bounceFreq + index * 0.3)) * bounceHeight;
            
            // 挤压效果
            const squish = isCurrentlyReading ? 1 + Math.sin(frame * bounceFreq * 2 + index * 0.3) * 0.1 : 1;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? "white"
                    : hasBeenRead 
                      ? "rgba(255, 255, 255, 0.9)"
                      : "white",
                  
                  // 弹跳球背景
                  background: isCurrentlyReading 
                    ? `radial-gradient(circle at 30% 30%, 
                        ${bounceColor}FF 0%, 
                        ${bounceColor}E6 30%, 
                        ${bounceColor}CC 70%, 
                        ${bounceColor}99 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "50px" : "0px",
                  padding: isCurrentlyReading ? "12px 20px" : "0px",
                  margin: isCurrentlyReading ? "0 6px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-6px" : "0px",
                  
                  // 3D弹跳球边框
                  border: isCurrentlyReading 
                    ? `3px solid ${bounceColor}40` 
                    : "none",
                  
                  // 弹跳变换
                  transform: isCurrentlyReading 
                    ? `scale(${1.2 * squish}, ${1.2 / Math.max(squish, 0.8)}) translateY(${-bounceY}px)`
                    : notYetRead 
                      ? "scale(0.7) translateY(20px)"
                      : "scale(1) translateY(0px)",
                  
                  transition: "all 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55)",
                  
                  // 弹跳球阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 ${Math.max(5, bounceY)}px ${20 + bounceY}px ${bounceColor}60,
                       inset 3px 3px 8px rgba(255, 255, 255, 0.4),
                       inset -2px -2px 6px rgba(0, 0, 0, 0.2),
                       0 ${bounceY + 10}px ${30 + bounceY * 2}px rgba(0, 0, 0, 0.3)`
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `2px 2px 0px ${bounceColor}80,
                       3px 3px 8px rgba(0, 0, 0, 0.6)`
                    : "2px 2px 6px rgba(0, 0, 0, 0.5)",
                  
                  // 高光效果
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "15%",
                    left: "20%",
                    width: "25%",
                    height: "25%",
                    background: "rgba(255, 255, 255, 0.8)",
                    borderRadius: "50%",
                    filter: "blur(3px)",
                    transform: `scale(${1 / squish})`,
                  } : undefined,
                  
                  // 弹跳轨迹
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: `${-bounceY - 15}px`,
                    left: "50%",
                    transform: "translateX(-50%)",
                    width: `${Math.max(20, 60 - bounceY)}px`,
                    height: `${height * 0.002}px`,
                    background: `radial-gradient(ellipse, ${bounceColor}40, transparent)`,
                    borderRadius: "50%",
                    filter: "blur(2px)",
                  } : undefined,
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 弹跳粒子效果 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`bounce-particle-${i}`}
          style={{
            position: "absolute",
            width: `${width * 0.004}px`,
            height: `${height * 0.002}px`,
            background: BOUNCE_COLORS[i % BOUNCE_COLORS.length],
            borderRadius: "50%",
            left: `${20 + i * 25}%`,
            bottom: `${300 + Math.abs(Math.sin(frame * 0.2 + i)) * 30}px`,
            opacity: Math.sin(frame * 0.1 + i) * 0.5 + 0.5,
            transform: `scale(${0.5 + Math.sin(frame * 0.15 + i) * 0.5})`,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
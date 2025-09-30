// 旋转螺旋样式 - 动态旋转螺旋效果
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




const SPIRAL_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA726", "#AB47BC", "#66BB6A"];

export const SpinSpiralStyle: React.FC<{
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
      {/* 螺旋背景装饰 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`spiral-${i}`}
          style={{
            position: "absolute",
            width: `${width * (0.046 + i * 0.028)}px`,
            height: `${height * (0.026 + i * 0.016)}px`,
            border: `2px solid ${SPIRAL_COLORS[i % SPIRAL_COLORS.length]}40`,
            borderRadius: "50%",
            left: "50%",
            top: "50%",
            transform: `translate(-50%, -50%) rotate(${frame * (2 + i * 0.5)}deg) scale(${0.5 + Math.sin(frame * 0.02) * 0.3})`,
            borderTopColor: `${SPIRAL_COLORS[i % SPIRAL_COLORS.length]}80`,
            borderRightColor: `${SPIRAL_COLORS[i % SPIRAL_COLORS.length]}60`,
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
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [25, 0])),
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

            const spiralColor = SPIRAL_COLORS[index % SPIRAL_COLORS.length];
            
            // 螺旋旋转计算
            const rotationSpeed = isCurrentlyReading ? 3 : 0;
            const rotation = frame * rotationSpeed + index * 45;
            
            // 螺旋轨道运动
            const orbitRadius = isCurrentlyReading ? 3 : 0;
            const orbitX = Math.cos((frame * 0.1 + index) * Math.PI / 180) * orbitRadius;
            const orbitY = Math.sin((frame * 0.1 + index) * Math.PI / 180) * orbitRadius;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? spiralColor
                    : hasBeenRead 
                      ? "rgba(255, 255, 255, 0.8)"
                      : "white",
                  
                  // 螺旋背景
                  background: isCurrentlyReading 
                    ? `conic-gradient(from ${rotation}deg, 
                        ${spiralColor}40 0deg, 
                        transparent 90deg, 
                        ${spiralColor}60 180deg, 
                        transparent 270deg, 
                        ${spiralColor}40 360deg)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "20px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 螺旋边框
                  border: isCurrentlyReading 
                    ? `2px solid ${spiralColor}80` 
                    : "none",
                  
                  transition: isCurrentlyReading ? "none" : "all 0.4s ease-out",
                  
                  // 螺旋变换
                  transform: isCurrentlyReading 
                    ? `scale(1.15) translateY(-6px) 
                       translateX(${orbitX}px) translateY(${orbitY}px)
                       rotate(${rotation}deg)`
                    : notYetRead 
                      ? "scale(0.85) translateY(8px) rotate(-15deg)"
                      : "scale(1) translateY(0px) rotate(0deg)",
                  
                  // 螺旋光晕
                  boxShadow: isCurrentlyReading 
                    ? `0 0 20px ${spiralColor}60,
                       0 0 40px ${spiralColor}40,
                       inset 0 0 20px ${spiralColor}20`
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 0 10px ${spiralColor},
                       2px 2px 8px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? `1px 1px 4px ${spiralColor}60`
                      : "2px 2px 6px rgba(0, 0, 0, 0.6)",
                  
                  // 螺旋装饰元素
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-15px",
                    left: "50%",
                    transform: `translateX(-50%) rotate(${rotation * 2}deg)`,
                    width: `${width * 0.007}px`,
                    height: `${height * 0.004}px`,
                    background: spiralColor,
                    borderRadius: "50%",
                    boxShadow: `0 0 10px ${spiralColor}`,
                  } : undefined,
                  
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: "-15px",
                    right: "20%",
                    transform: `rotate(${-rotation * 1.5}deg)`,
                    width: `${width * 0.006}px`,
                    height: `${height * 0.003}px`,
                    background: `${spiralColor}80`,
                    borderRadius: "50%",
                  } : undefined,
                }}
              >
                {/* 旋转轨迹 */}
                {isCurrentlyReading && (
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      width: `${width * 0.056}px`,
                      height: `${height * 0.031}px`,
                      border: `1px dashed ${spiralColor}40`,
                      borderRadius: "50%",
                      transform: `translate(-50%, -50%) rotate(${-rotation}deg)`,
                      pointerEvents: "none",
                    }}
                  />
                )}
                
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 螺旋粒子轨迹 */}
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={`spiral-particle-${i}`}
          style={{
            position: "absolute",
            width: `${width * 0.003}px`,
            height: `${height * 0.002}px`,
            background: SPIRAL_COLORS[i % SPIRAL_COLORS.length],
            borderRadius: "50%",
            left: `${50 + Math.cos(frame * 0.05 + i * 72) * (20 + i * 5)}%`,
            top: `${50 + Math.sin(frame * 0.05 + i * 72) * (15 + i * 3)}%`,
            opacity: Math.sin(frame * 0.1 + i) * 0.4 + 0.6,
            transform: `scale(${0.5 + Math.sin(frame * 0.08 + i) * 0.5})`,
            filter: `hue-rotate(${frame * 2}deg)`,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
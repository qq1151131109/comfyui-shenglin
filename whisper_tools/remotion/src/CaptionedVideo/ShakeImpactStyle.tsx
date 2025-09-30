// 震动冲击样式 - 强烈震动冲击效果
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  random,
} from "remotion";
import { TheBoldFont } from "../load-font";
import { fitText } from "@remotion/layout-utils";
import { makeTransform, scale, translateY } from "@remotion/animation-utils";
import { TikTokPage } from "@remotion/captions";

const fontFamily = TheBoldFont;




const IMPACT_COLORS = ["#FF3030", "#FFD700", "#FF6B35", "#FF1744", "#FF4081", "#FFC107"];

export const ShakeImpactStyle: React.FC<{
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
            scale(interpolate(enterProgress, [0, 1], [0.85, 1])),
            translateY(interpolate(enterProgress, [0, 1], [35, 0])),
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

            const impactColor = IMPACT_COLORS[index % IMPACT_COLORS.length];
            
            // 震动强度计算
            const shakeIntensity = isCurrentlyReading ? 4 : 0;
            const shakeX = isCurrentlyReading ? random(`shake-x-${t.fromMs}-${Math.floor(frame / 2)}`) * shakeIntensity - shakeIntensity / 2 : 0;
            const shakeY = isCurrentlyReading ? random(`shake-y-${t.fromMs}-${Math.floor(frame / 2)}`) * shakeIntensity - shakeIntensity / 2 : 0;
            
            // 冲击波效果
            const impactPulse = isCurrentlyReading ? Math.abs(Math.sin(frame * 0.5)) : 0;
            const impactScale = 1 + impactPulse * 0.15;
            
            // 颜色闪烁
            const colorFlicker = isCurrentlyReading && frame % 4 < 2;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? (colorFlicker ? "#FFFFFF" : impactColor)
                    : hasBeenRead 
                      ? `${impactColor}CC`
                      : "white",
                  
                  // 冲击背景
                  background: isCurrentlyReading 
                    ? `radial-gradient(circle at center, 
                        ${impactColor}${colorFlicker ? 'FF' : 'E6'} 0%, 
                        ${impactColor}${colorFlicker ? 'E6' : 'CC'} 30%, 
                        ${impactColor}${colorFlicker ? 'CC' : '99'} 70%, 
                        ${impactColor}${colorFlicker ? '99' : '66'} 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "15px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 冲击边框
                  border: isCurrentlyReading 
                    ? `3px solid ${colorFlicker ? '#FFFFFF' : impactColor}` 
                    : "none",
                  
                  transition: colorFlicker ? "none" : "all 0.1s ease-out",
                  
                  // 震动变换
                  transform: isCurrentlyReading 
                    ? `scale(${1.1 * impactScale}) 
                       translateY(${-6 + shakeY}px) 
                       translateX(${shakeX}px)
                       ${colorFlicker ? `skew(${random(`skew-${t.fromMs}-${frame}`) * 4 - 2}deg)` : ''}`
                    : notYetRead 
                      ? "scale(0.85) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  // 冲击阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 ${8 + Math.abs(shakeY)}px ${20 + impactPulse * 20}px ${impactColor}80,
                       inset 0 0 20px ${colorFlicker ? '#FFFFFF40' : `${impactColor}40`},
                       0 0 40px ${impactColor}60,
                       ${colorFlicker ? `0 0 60px #FFFFFF60` : ''}`
                    : "none",
                  
                  // 震动文字阴影
                  textShadow: isCurrentlyReading 
                    ? `${2 + shakeX * 0.5}px ${2 + shakeY * 0.5}px 0px ${impactColor}${colorFlicker ? '40' : '80'},
                       ${4 + shakeX}px ${4 + shakeY}px 8px rgba(0, 0, 0, 0.8),
                       ${colorFlicker ? `0 0 20px #FFFFFF80` : `0 0 15px ${impactColor}60`}`
                    : hasBeenRead
                      ? `1px 1px 4px ${impactColor}60`
                      : "2px 2px 6px rgba(0, 0, 0, 0.5)",
                  
                  // 字体粗细
                  fontWeight: isCurrentlyReading ? "900" : "700",
                  
                  // 冲击装饰
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: `${-8 + shakeY * 0.3}px`,
                    left: `${-8 + shakeX * 0.3}px`,
                    width: `${6 + impactPulse * 4}px`,
                    height: `${6 + impactPulse * 4}px`,
                    background: colorFlicker ? "#FFFFFF" : impactColor,
                    transform: `rotate(${random(`rotate1-${t.fromMs}-${Math.floor(frame / 3)}`) * 360}deg)`,
                    opacity: 0.8,
                  } : undefined,
                  
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: `${-8 - shakeY * 0.3}px`,
                    right: `${-8 - shakeX * 0.3}px`,
                    width: `${4 + impactPulse * 3}px`,
                    height: `${4 + impactPulse * 3}px`,
                    background: colorFlicker ? "#FFFFFF80" : `${impactColor}80`,
                    transform: `rotate(${random(`rotate2-${t.fromMs}-${Math.floor(frame / 3)}`) * 360}deg)`,
                    opacity: 0.6,
                  } : undefined,
                }}
              >
                {/* 冲击波环 */}
                {isCurrentlyReading && (
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      width: `${60 + impactPulse * 40}px`,
                      height: `${60 + impactPulse * 40}px`,
                      border: `2px solid ${impactColor}${Math.floor((1 - impactPulse) * 80).toString(16)}`,
                      borderRadius: "50%",
                      transform: "translate(-50%, -50%)",
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
      
      {/* 冲击碎片效果 */}
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div
          key={`impact-debris-${i}`}
          style={{
            position: "absolute",
            width: `${2 + i}px`,
            height: `${2 + i}px`,
            background: IMPACT_COLORS[i % IMPACT_COLORS.length],
            left: `${20 + i * 12 + random(`debris-x-${i}-${Math.floor(frame / 5)}`) * 20}%`,
            top: `${40 + random(`debris-y-${i}-${Math.floor(frame / 5)}`) * 30}%`,
            transform: `rotate(${random(`debris-rot-${i}-${frame}`) * 360}deg) scale(${0.5 + random(`debris-scale-${i}-${Math.floor(frame / 3)}`) * 1.5})`,
            opacity: random(`debris-op-${i}-${Math.floor(frame / 8)}`) * 0.8 + 0.2,
          }}
        />
      ))}
      
      {/* 震动线条 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`shake-line-${i}`}
          style={{
            position: "absolute",
            width: "100%",
            height: `${height * 0.001}px`,
            background: `linear-gradient(90deg, transparent, ${IMPACT_COLORS[i % IMPACT_COLORS.length]}60, transparent)`,
            top: `${45 + i * 8 + random(`line-${i}-${Math.floor(frame / 2)}`) * 10}%`,
            opacity: random(`line-op-${i}-${Math.floor(frame / 4)}`) * 0.7 + 0.3,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
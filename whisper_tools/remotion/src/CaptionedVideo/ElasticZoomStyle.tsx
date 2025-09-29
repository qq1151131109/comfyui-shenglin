// 弹性缩放样式 - 有趣的弹性拉伸效果
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
const ELASTIC_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"];

export const ElasticZoomStyle: React.FC<{
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
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.85, 1])),
            translateY(interpolate(enterProgress, [0, 1], [30, 0])),
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

            const elasticColor = ELASTIC_COLORS[index % ELASTIC_COLORS.length];
            
            // 弹性动画计算
            const elasticPhase = (frame * 0.15 + index * 0.3) % (Math.PI * 2);
            const elasticStretchX = isCurrentlyReading ? 1 + Math.sin(elasticPhase) * 0.15 : 1;
            const elasticStretchY = isCurrentlyReading ? 1 + Math.cos(elasticPhase * 1.3) * 0.1 : 1;
            
            // 弹性位移
            const elasticOffsetX = isCurrentlyReading ? Math.sin(elasticPhase * 0.7) * 2 : 0;
            const elasticOffsetY = isCurrentlyReading ? Math.cos(elasticPhase * 0.5) * 3 : 0;

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
                      ? elasticColor
                      : "white",
                  
                  // 弹性背景
                  background: isCurrentlyReading 
                    ? `radial-gradient(ellipse ${100 * elasticStretchX}% ${100 * elasticStretchY}% at center, 
                        ${elasticColor}E6 0%, 
                        ${elasticColor}CC 40%, 
                        ${elasticColor}99 70%, 
                        ${elasticColor}66 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "25px" : "0px",
                  padding: isCurrentlyReading ? "10px 18px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-5px" : "0px",
                  
                  // 弹性边框
                  border: isCurrentlyReading 
                    ? `3px solid ${elasticColor}40` 
                    : "none",
                  
                  transition: "color 0.3s ease, border 0.3s ease",
                  
                  // 弹性变换
                  transform: isCurrentlyReading 
                    ? `scale(${1.1 * elasticStretchX}, ${1.1 * elasticStretchY}) 
                       translateY(${-8 + elasticOffsetY}px) 
                       translateX(${elasticOffsetX}px)
                       perspective(500px) rotateX(${Math.sin(elasticPhase) * 3}deg)`
                    : notYetRead 
                      ? "scale(0.8) translateY(10px)"
                      : "scale(1) translateY(0px)",
                  
                  // 弹性阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 ${8 + Math.abs(elasticOffsetY)}px ${20 + Math.abs(elasticOffsetY * 2)}px ${elasticColor}60,
                       inset 0 0 20px ${elasticColor}20,
                       0 0 40px ${elasticColor}40`
                    : hasBeenRead
                      ? `0 2px 8px ${elasticColor}40`
                      : "0 2px 6px rgba(0, 0, 0, 0.3)",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `2px 2px 0px ${elasticColor}80,
                       3px 3px 8px rgba(0, 0, 0, 0.7),
                       0 0 15px ${elasticColor}60`
                    : hasBeenRead
                      ? `1px 1px 4px ${elasticColor}60`
                      : "2px 2px 6px rgba(0, 0, 0, 0.5)",
                  
                  // 弹性轮廓
                  outline: isCurrentlyReading 
                    ? `1px solid ${elasticColor}60` 
                    : "none",
                  outlineOffset: isCurrentlyReading ? `${2 + Math.abs(elasticOffsetX)}px` : "0px",
                  
                  // 弹性装饰点
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: `${-5 - elasticOffsetY}px`,
                    left: `${-5 - elasticOffsetX}px`,
                    width: `${6 + Math.abs(elasticOffsetX)}px`,
                    height: `${6 + Math.abs(elasticOffsetY)}px`,
                    background: elasticColor,
                    borderRadius: "50%",
                    transform: `scale(${elasticStretchX * 0.8}, ${elasticStretchY * 0.8})`,
                    opacity: 0.8,
                  } : undefined,
                  
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: `${-5 + elasticOffsetY}px`,
                    right: `${-5 + elasticOffsetX}px`,
                    width: `${4 + Math.abs(elasticOffsetY)}px`,
                    height: `${4 + Math.abs(elasticOffsetX)}px`,
                    background: `${elasticColor}80`,
                    borderRadius: "50%",
                    transform: `scale(${elasticStretchY * 0.6}, ${elasticStretchX * 0.6})`,
                    opacity: 0.6,
                  } : undefined,
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 弹性波纹效果 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`elastic-ripple-${i}`}
          style={{
            position: "absolute",
            width: `${30 + i * 20}px`,
            height: `${20 + i * 10}px`,
            border: `2px solid ${ELASTIC_COLORS[i % ELASTIC_COLORS.length]}40`,
            borderRadius: "50%",
            left: `${30 + i * 20}%`,
            top: `${40 + Math.sin(frame * 0.05 + i) * 15}%`,
            transform: `scale(${1 + Math.sin(frame * 0.08 + i) * 0.3}, ${1 + Math.cos(frame * 0.06 + i) * 0.2})`,
            opacity: Math.abs(Math.sin(frame * 0.04 + i)) * 0.6 + 0.2,
          }}
        />
      ))}
      
      {/* 弹性粒子 */}
      {[1, 2, 3, 4].map((i) => (
        <div
          key={`elastic-particle-${i}`}
          style={{
            position: "absolute",
            width: "5px",
            height: "5px",
            background: ELASTIC_COLORS[i % ELASTIC_COLORS.length],
            borderRadius: "50%",
            left: `${15 + i * 20}%`,
            top: `${45 + Math.sin(frame * 0.03 + i * 1.2) * 20}%`,
            transform: `scale(${0.3 + Math.abs(Math.sin(frame * 0.1 + i))})`,
            opacity: Math.sin(frame * 0.07 + i) * 0.4 + 0.6,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
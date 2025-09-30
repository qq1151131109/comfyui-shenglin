// 水波涟漪样式 - 清澈水波涟漪效果
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





export const WaterRippleStyle: React.FC<{
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

            // 水波动画
            const wavePhase = (frame * 0.1 + index * 0.4) % (Math.PI * 2);
            const rippleIntensity = Math.sin(wavePhase) * 0.5 + 0.5;
            
            // 水的渐变色
            const waterGradient = `linear-gradient(45deg,
              #87CEEB 0%,
              #4682B4 25%,
              #1E90FF 50%,
              #00BFFF 75%,
              #87CEFA 100%)`;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 水波文字渐变
                  background: isCurrentlyReading 
                    ? waterGradient
                    : hasBeenRead
                      ? `linear-gradient(45deg, #87CEEB 0%, #4682B4 100%)`
                      : `linear-gradient(45deg, #FFFFFF 0%, #F0F8FF 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundSize: isCurrentlyReading ? "200% 200%" : "100% 100%",
                  
                  // 水波背景层
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-8px",
                    left: "-12px",
                    right: "-12px",
                    bottom: "-8px",
                    background: `radial-gradient(ellipse at center,
                      rgba(135, 206, 235, 0.3) 0%,
                      rgba(70, 130, 180, 0.2) 40%,
                      rgba(30, 144, 255, 0.1) 70%,
                      transparent 100%)`,
                    borderRadius: "20px",
                    filter: "blur(4px)",
                    zIndex: -1,
                  } : undefined,
                  
                  borderRadius: isCurrentlyReading ? "16px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 水波边框
                  border: isCurrentlyReading 
                    ? `2px solid rgba(135, 206, 235, ${0.6 + rippleIntensity * 0.4})` 
                    : "none",
                  
                  // 水波动画
                  animation: isCurrentlyReading 
                    ? "water-flow 3s ease-in-out infinite" 
                    : "none",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.08) translateY(${-6 + Math.sin(wavePhase) * 1.5}px)`
                    : notYetRead 
                      ? "scale(0.92) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 水波光效
                  boxShadow: isCurrentlyReading 
                    ? `0 4px 20px rgba(135, 206, 235, 0.6),
                       inset 0 2px 4px rgba(255, 255, 255, 0.3),
                       inset 0 -2px 4px rgba(70, 130, 180, 0.3),
                       0 0 30px rgba(30, 144, 255, 0.4)`
                    : hasBeenRead
                      ? "0 2px 8px rgba(135, 206, 235, 0.4)"
                      : "0 1px 3px rgba(0, 0, 0, 0.2)",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 2px 4px rgba(70, 130, 180, 0.6),
                       0 0 10px rgba(135, 206, 235, 0.8)`
                    : hasBeenRead
                      ? "0 1px 3px rgba(135, 206, 235, 0.6)"
                      : "0 1px 2px rgba(0, 0, 0, 0.3)",
                  
                  transition: "all 0.4s ease-out",
                  
                  // 水滴装饰
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-6px",
                    right: "10%",
                    width: `${width * 0.004}px`,
                    height: `${height * 0.003}px`,
                    background: "rgba(135, 206, 235, 0.8)",
                    borderRadius: "50% 50% 50% 50% / 40% 40% 60% 60%",
                    animation: "water-drop 2s ease-in-out infinite",
                  } : undefined,
                }}
              >
                {/* 涟漪扩散效果 */}
                {isCurrentlyReading && [1, 2, 3].map((i) => (
                  <div
                    key={`ripple-${i}`}
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      width: `${width * (0.028 + i * 0.019)}px`,
                      height: `${height * (0.010 + i * 0.008)}px`,
                      border: `1px solid rgba(135, 206, 235, ${(1 - i * 0.3) * (0.5 + rippleIntensity * 0.5)})`,
                      borderRadius: "50%",
                      transform: `translate(-50%, -50%) scale(${0.5 + rippleIntensity * (1 + i * 0.3)})`,
                      animation: `ripple-expand-${i} 2s ease-out infinite`,
                      pointerEvents: "none",
                    }}
                  />
                ))}
                
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 水泡效果 */}
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={`water-bubble-${i}`}
          style={{
            position: "absolute",
            width: `${width * (0.003 + i * 0.002)}px`,
            height: `${height * (0.002 + i * 0.001)}px`,
            background: `radial-gradient(circle at 30% 30%, 
              rgba(255, 255, 255, 0.8) 0%, 
              rgba(135, 206, 235, 0.6) 50%, 
              rgba(70, 130, 180, 0.4) 100%)`,
            borderRadius: "50%",
            left: `${20 + i * 15}%`,
            bottom: `${250 + Math.sin(frame * 0.05 + i * 0.8) * 30}px`,
            opacity: Math.sin(frame * 0.03 + i * 0.5) * 0.4 + 0.6,
            transform: `scale(${0.5 + Math.sin(frame * 0.04 + i) * 0.3})`,
            filter: "blur(1px)",
          }}
        />
      ))}
      
      {/* 水面反光 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`water-reflection-${i}`}
          style={{
            position: "absolute",
            width: `${width * (0.046 + i * 0.028)}px`,
            height: `${height * 0.001}px`,
            background: `linear-gradient(90deg, 
              transparent, 
              rgba(255, 255, 255, ${0.3 + i * 0.1}), 
              transparent)`,
            left: `${10 + i * 25}%`,
            top: `${55 + Math.sin(frame * 0.08 + i * 0.6) * 8}%`,
            opacity: Math.sin(frame * 0.06 + i * 0.4) * 0.5 + 0.5,
            borderRadius: "1px",
            filter: "blur(1px)",
            transform: `scaleX(${0.8 + Math.sin(frame * 0.04 + i) * 0.2})`,
          }}
        />
      ))}
      
      <style>{`
        @keyframes water-flow {
          0%, 100% { 
            background-position: 0% 50%;
            border-color: rgba(135, 206, 235, 0.6);
          }
          50% { 
            background-position: 100% 50%;
            border-color: rgba(30, 144, 255, 0.8);
          }
        }
        
        @keyframes water-drop {
          0%, 100% { 
            transform: translateY(0px) scale(1);
            opacity: 1;
          }
          50% { 
            transform: translateY(3px) scale(0.8);
            opacity: 0.6;
          }
        }
        
        @keyframes ripple-expand-1 {
          0% { 
            transform: translate(-50%, -50%) scale(0.3);
            opacity: 0.8;
          }
          100% { 
            transform: translate(-50%, -50%) scale(2);
            opacity: 0;
          }
        }
        
        @keyframes ripple-expand-2 {
          0% { 
            transform: translate(-50%, -50%) scale(0.5);
            opacity: 0.6;
          }
          100% { 
            transform: translate(-50%, -50%) scale(2.5);
            opacity: 0;
          }
        }
        
        @keyframes ripple-expand-3 {
          0% { 
            transform: translate(-50%, -50%) scale(0.7);
            opacity: 0.4;
          }
          100% { 
            transform: translate(-50%, -50%) scale(3);
            opacity: 0;
          }
        }
      `}</style>
    </AbsoluteFill>
  );
};
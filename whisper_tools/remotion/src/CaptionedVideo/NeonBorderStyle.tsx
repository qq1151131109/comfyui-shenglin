// 霓虹边框样式 - 炫酷霓虹边框效果
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




const NEON_COLORS = ["#00F5FF", "#FF1493", "#00FF41", "#FFD700", "#FF4500", "#DA70D6"];

export const NeonBorderStyle: React.FC<{
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

            const neonColor = NEON_COLORS[index % NEON_COLORS.length];
            const pulseIntensity = Math.sin(frame * 0.15) * 0.5 + 0.5;

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
                      ? "rgba(255, 255, 255, 0.8)"
                      : "white",
                  
                  // 霓虹边框
                  border: isCurrentlyReading 
                    ? `3px solid ${neonColor}` 
                    : hasBeenRead
                      ? `1px solid ${neonColor}80`
                      : "none",
                  borderRadius: isCurrentlyReading ? "12px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 背景
                  backgroundColor: isCurrentlyReading 
                    ? `${neonColor}15` 
                    : "transparent",
                  
                  transition: "all 0.3s ease-out",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.1) translateY(-6px)`
                    : notYetRead 
                      ? "scale(0.9) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  // 多层霓虹发光
                  boxShadow: isCurrentlyReading 
                    ? `0 0 5px ${neonColor},
                       0 0 10px ${neonColor},
                       0 0 15px ${neonColor},
                       0 0 20px ${neonColor},
                       inset 0 0 5px ${neonColor}${Math.floor(30 + pulseIntensity * 20).toString(16)}`
                    : hasBeenRead
                      ? `0 0 3px ${neonColor}80`
                      : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 0 5px ${neonColor},
                       0 0 10px ${neonColor},
                       2px 2px 8px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? `0 0 3px ${neonColor}60`
                      : "2px 2px 6px rgba(0, 0, 0, 0.6)",
                  
                  // 动态边框动画
                  animation: isCurrentlyReading 
                    ? `neon-pulse-${index % 3} 2s ease-in-out infinite` 
                    : "none",
                  
                  // 伪元素创建额外发光层
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-6px",
                    left: "-6px",
                    right: "-6px", 
                    bottom: "-6px",
                    background: "transparent",
                    border: `1px solid ${neonColor}40`,
                    borderRadius: "16px",
                    animation: `neon-orbit 3s linear infinite`,
                  } : undefined,
                  
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-10px",
                    left: "-10px",
                    right: "-10px",
                    bottom: "-10px", 
                    background: "transparent",
                    border: `1px solid ${neonColor}20`,
                    borderRadius: "20px",
                    animation: `neon-orbit 4s linear infinite reverse`,
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
        @keyframes neon-pulse-0 {
          0%, 100% { 
            filter: brightness(1) saturate(1);
            box-shadow: 
              0 0 5px currentColor,
              0 0 10px currentColor,
              0 0 15px currentColor,
              0 0 20px currentColor;
          }
          50% { 
            filter: brightness(1.5) saturate(1.5);
            box-shadow: 
              0 0 10px currentColor,
              0 0 20px currentColor,
              0 0 30px currentColor,
              0 0 40px currentColor;
          }
        }
        
        @keyframes neon-pulse-1 {
          0%, 100% { 
            filter: brightness(1) saturate(1);
          }
          25% { 
            filter: brightness(1.3) saturate(1.3);
          }
          75% { 
            filter: brightness(1.2) saturate(1.2);
          }
        }
        
        @keyframes neon-pulse-2 {
          0%, 100% { 
            filter: brightness(1) saturate(1) hue-rotate(0deg);
          }
          33% { 
            filter: brightness(1.4) saturate(1.4) hue-rotate(30deg);
          }
          66% { 
            filter: brightness(1.2) saturate(1.2) hue-rotate(-30deg);
          }
        }
        
        @keyframes neon-orbit {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </AbsoluteFill>
  );
};
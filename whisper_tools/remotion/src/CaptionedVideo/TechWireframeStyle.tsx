// 科技线框样式 - 未来科技HUD效果
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
const TECH_COLORS = ["#00D4FF", "#00FF88", "#FF6B00", "#FF0080", "#8000FF", "#FFFF00"];

export const TechWireframeStyle: React.FC<{
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
      {/* 科技网格背景 */}
      <div
        style={{
          position: "absolute",
          width: "120%",
          height: "120%",
          backgroundImage: `
            linear-gradient(rgba(0, 212, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
          opacity: 0.6,
          animation: "grid-move 10s linear infinite",
        }}
      />
      
      {/* 扫描线效果 */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "2px",
          background: `linear-gradient(90deg, transparent, #00D4FF, transparent)`,
          top: `${50 + Math.sin(frame * 0.03) * 20}%`,
          opacity: 0.7,
          animation: "scan-line 3s ease-in-out infinite",
        }}
      />
      
      <div
        style={{
          fontSize,
          color: "#E0E0E0",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [20, 0])),
          ]),
          filter: "drop-shadow(0 0 10px rgba(0, 212, 255, 0.3))",
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

            const techColor = TECH_COLORS[index % TECH_COLORS.length];

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? techColor
                    : hasBeenRead 
                      ? "#B0B0B0"
                      : "#E0E0E0",
                  
                  // 科技边框
                  border: isCurrentlyReading 
                    ? `2px solid ${techColor}` 
                    : hasBeenRead
                      ? `1px solid ${techColor}40`
                      : "1px solid rgba(224, 224, 224, 0.2)",
                  borderRadius: "0px", // 保持方形科技感
                  padding: isCurrentlyReading ? "8px 14px" : "2px 4px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 科技背景
                  background: isCurrentlyReading 
                    ? `linear-gradient(135deg, 
                        ${techColor}15 0%, 
                        transparent 50%, 
                        ${techColor}10 100%)`
                    : "rgba(0, 0, 0, 0.2)",
                  
                  transition: "all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.1) translateY(-6px) perspective(500px) rotateX(${Math.sin(frame * 0.1) * 2}deg)`
                    : notYetRead 
                      ? "scale(0.9) translateY(4px)"
                      : "scale(1) translateY(0px)",
                  
                  // 科技发光
                  boxShadow: isCurrentlyReading 
                    ? `0 0 20px ${techColor}80,
                       inset 0 0 10px ${techColor}20,
                       0 4px 8px rgba(0, 0, 0, 0.3)`
                    : hasBeenRead
                      ? `0 0 5px ${techColor}40`
                      : "0 1px 3px rgba(0, 0, 0, 0.3)",
                  
                  // 文字发光效果
                  textShadow: isCurrentlyReading 
                    ? `0 0 10px ${techColor},
                       0 0 20px ${techColor}80,
                       1px 1px 2px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? `0 0 5px ${techColor}60`
                      : "1px 1px 2px rgba(0, 0, 0, 0.5)",
                  
                  // 科技装饰元素
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-4px",
                    left: "-4px",
                    width: "8px",
                    height: "8px",
                    border: `2px solid ${techColor}`,
                    background: techColor,
                    animation: "tech-blink 1s ease-in-out infinite",
                  } : undefined,
                  
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: "-4px",
                    right: "-4px",
                    width: "8px",
                    height: "8px",
                    border: `2px solid ${techColor}`,
                    background: "transparent",
                    animation: "tech-blink 1s ease-in-out infinite 0.5s",
                  } : undefined,
                }}
              >
                {/* HUD数据显示 */}
                {isCurrentlyReading && (
                  <div
                    style={{
                      position: "absolute",
                      top: "-25px",
                      left: "50%",
                      transform: "translateX(-50%)",
                      fontSize: "10px",
                      color: techColor,
                      fontFamily: "monospace",
                      whiteSpace: "nowrap",
                      opacity: 0.8,
                    }}
                  >
                    [{String(Math.floor(timeInMs)).padStart(4, '0')}ms]
                  </div>
                )}
                
                {t.text}
                
                {/* 进度条 */}
                {isCurrentlyReading && (
                  <div
                    style={{
                      position: "absolute",
                      bottom: "-8px",
                      left: "0",
                      right: "0",
                      height: "2px",
                      background: "rgba(255, 255, 255, 0.2)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        background: techColor,
                        width: `${((timeInMs - startRelativeToSequence) / (endRelativeToSequence - startRelativeToSequence)) * 100}%`,
                        transition: "width 0.1s linear",
                      }}
                    />
                  </div>
                )}
              </span>
            );
          })}
        </span>
      </div>
      
      <style>{`
        @keyframes grid-move {
          0% { transform: translate(0, 0); }
          100% { transform: translate(20px, 20px); }
        }
        
        @keyframes scan-line {
          0%, 100% { opacity: 0.7; transform: scaleX(0.8); }
          50% { opacity: 1; transform: scaleX(1.2); }
        }
        
        @keyframes tech-blink {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.8); }
        }
      `}</style>
    </AbsoluteFill>
  );
};
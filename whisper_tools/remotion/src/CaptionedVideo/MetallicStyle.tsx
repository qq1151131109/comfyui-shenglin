// 金属质感样式 - 豪华金属效果
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





export const MetallicStyle: React.FC<{
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

  // 光泽动画位置
  const shinePosition = (frame * 2) % 200;

  return (
    <AbsoluteFill style={container}>
      {/* 背景金属质感 */}
      <div
        style={{
          position: "absolute",
          width: "120%",
          height: "200%",
          background: `radial-gradient(ellipse at center, 
            rgba(212, 175, 55, 0.1) 0%,
            rgba(184, 134, 11, 0.05) 50%,
            transparent 100%)`,
          filter: "blur(20px)",
        }}
      />
      
      <div
        style={{
          fontSize,
          color: "transparent",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [20, 0])),
          ]),
          // 基础金属渐变
          background: `linear-gradient(45deg,
            #D4AF37 0%,
            #FFD700 25%, 
            #FFF8DC 50%, 
            #FFD700 75%, 
            #B8860B 100%)`,
          WebkitBackgroundClip: "text",
          backgroundSize: "200% 200%",
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

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 金属文字渐变
                  background: isCurrentlyReading 
                    ? `linear-gradient(45deg,
                        #C0C0C0 0%,
                        #FFFFFF 20%,
                        #E6E6FA 40%,
                        #FFFFFF 60%,
                        #C0C0C0 80%,
                        #A9A9A9 100%)`
                    : hasBeenRead
                      ? `linear-gradient(45deg,
                          #D4AF37 0%,
                          #FFD700 50%, 
                          #B8860B 100%)`
                      : `linear-gradient(45deg,
                          #708090 0%,
                          #C0C0C0 50%, 
                          #708090 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundSize: isCurrentlyReading ? "300% 300%" : "200% 200%",
                  
                  // 金属背景框
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-4px",
                    left: "-8px",
                    right: "-8px",
                    bottom: "-4px",
                    background: `linear-gradient(135deg,
                      rgba(192, 192, 192, 0.3) 0%,
                      rgba(255, 255, 255, 0.1) 50%,
                      rgba(169, 169, 169, 0.3) 100%)`,
                    borderRadius: "12px",
                    border: "2px solid rgba(255, 255, 255, 0.4)",
                    zIndex: -1,
                  } : undefined,
                  
                  padding: isCurrentlyReading ? "6px 14px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  transition: "all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.1) translateY(-4px) perspective(500px) rotateX(${Math.sin(frame * 0.05) * 3}deg)`
                    : notYetRead 
                      ? "scale(0.9) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 金属光泽
                  boxShadow: isCurrentlyReading 
                    ? `0 6px 20px rgba(192, 192, 192, 0.4),
                       inset 0 1px 0 rgba(255, 255, 255, 0.6),
                       inset 0 -1px 0 rgba(0, 0, 0, 0.2),
                       0 0 30px rgba(255, 255, 255, 0.3)`
                    : hasBeenRead
                      ? `0 2px 8px rgba(212, 175, 55, 0.3)`
                      : "0 1px 3px rgba(0, 0, 0, 0.3)",
                  
                  // 文字阴影增强金属感
                  textShadow: isCurrentlyReading 
                    ? `1px 1px 0px rgba(0, 0, 0, 0.4),
                       2px 2px 0px rgba(0, 0, 0, 0.3),
                       3px 3px 0px rgba(0, 0, 0, 0.2),
                       0 0 10px rgba(255, 255, 255, 0.6)`
                    : hasBeenRead
                      ? `1px 1px 0px rgba(0, 0, 0, 0.3),
                         0 0 5px rgba(212, 175, 55, 0.5)`
                      : "1px 1px 2px rgba(0, 0, 0, 0.5)",
                  
                  // 动态光泽扫过效果
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "0",
                    left: `${shinePosition - 100}%`,
                    width: "100%",
                    height: "100%",
                    background: `linear-gradient(90deg,
                      transparent 0%,
                      rgba(255, 255, 255, 0.8) 50%,
                      transparent 100%)`,
                    animation: "shine 3s ease-in-out infinite",
                    pointerEvents: "none",
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
        @keyframes shine {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </AbsoluteFill>
  );
};
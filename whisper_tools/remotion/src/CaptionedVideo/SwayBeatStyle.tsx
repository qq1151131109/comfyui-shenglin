// 摇摆节拍样式 - 音乐节拍摇摆效果
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




const BEAT_COLORS = ["#FF1744", "#00E676", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4"];

export const SwayBeatStyle: React.FC<{
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
      {/* 音乐节拍背景 */}
      {[1, 2, 3, 4].map((i) => (
        <div
          key={`beat-bar-${i}`}
          style={{
            position: "absolute",
            width: `${width * 0.004}px`,
            height: `${20 + Math.abs(Math.sin(frame * 0.2 + i * 0.5)) * 40}px`,
            background: `linear-gradient(to top, ${BEAT_COLORS[i % BEAT_COLORS.length]}, ${BEAT_COLORS[i % BEAT_COLORS.length]}80)`,
            left: `${15 + i * 15}%`,
            bottom: "200px",
            borderRadius: "2px",
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
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [20, 0])),
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

            const beatColor = BEAT_COLORS[index % BEAT_COLORS.length];
            
            // 节拍摇摆计算
            const beatFreq = 0.2; // 节拍频率
            const swayAngle = isCurrentlyReading ? Math.sin(frame * beatFreq + index * 0.4) * 8 : 0;
            const beatScale = isCurrentlyReading ? 1 + Math.abs(Math.sin(frame * beatFreq * 2 + index * 0.3)) * 0.1 : 1;
            
            // 节拍跳动
            const beatBounce = isCurrentlyReading ? Math.abs(Math.sin(frame * beatFreq * 4 + index * 0.2)) * 4 : 0;

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
                      ? beatColor
                      : "white",
                  
                  // 节拍背景
                  background: isCurrentlyReading 
                    ? `linear-gradient(135deg, 
                        ${beatColor}E6 0%, 
                        ${beatColor}B3 50%, 
                        ${beatColor}80 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "18px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 节拍边框
                  border: isCurrentlyReading 
                    ? `2px solid ${beatColor}60` 
                    : "none",
                  
                  transition: "all 0.2s ease-out",
                  
                  // 摇摆变换
                  transform: isCurrentlyReading 
                    ? `scale(${1.05 * beatScale}) 
                       translateY(${-6 - beatBounce}px) 
                       rotate(${swayAngle}deg)
                       perspective(400px) rotateY(${swayAngle * 0.5}deg)`
                    : notYetRead 
                      ? "scale(0.9) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 摇摆阴影
                  boxShadow: isCurrentlyReading 
                    ? `${Math.sin(swayAngle * Math.PI / 180) * 5}px ${8 + beatBounce}px ${15 + beatBounce}px ${beatColor}60,
                       inset 0 0 15px ${beatColor}30,
                       0 0 30px ${beatColor}40`
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `${Math.cos(swayAngle * Math.PI / 180) * 2}px ${2 + beatBounce * 0.5}px 8px ${beatColor}80,
                       ${Math.sin(swayAngle * Math.PI / 180) * 2}px ${2 + beatBounce * 0.5}px 15px rgba(0, 0, 0, 0.7)`
                    : hasBeenRead
                      ? `1px 1px 4px ${beatColor}60`
                      : "2px 2px 6px rgba(0, 0, 0, 0.5)",
                  
                  // 节拍装饰
                  '::before': isCurrentlyReading ? {
                    content: '"♪"',
                    position: "absolute",
                    top: "-20px",
                    left: `${Math.sin(swayAngle * Math.PI / 180) * 10 + 50}%`,
                    transform: `translateX(-50%) rotate(${-swayAngle}deg)`,
                    fontSize: "14px",
                    color: beatColor,
                    opacity: 0.8,
                    animation: `beat-note ${1 / beatFreq}ms ease-in-out infinite`,
                  } : undefined,
                  
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: "-10px",
                    left: "50%",
                    transform: "translateX(-50%)",
                    width: `${20 + beatScale * 10}px`,
                    height: `${height * 0.002}px`,
                    background: `linear-gradient(90deg, transparent, ${beatColor}60, transparent)`,
                    borderRadius: "2px",
                    opacity: 0.7,
                  } : undefined,
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 音符装饰 */}
      {["♪", "♫", "♬", "♭", "♯"].map((note, i) => (
        <div
          key={`music-note-${i}`}
          style={{
            position: "absolute",
            fontSize: `${12 + i * 2}px`,
            color: BEAT_COLORS[i % BEAT_COLORS.length],
            left: `${20 + i * 15}%`,
            top: `${30 + Math.sin(frame * 0.08 + i * 0.8) * 25}%`,
            opacity: Math.sin(frame * 0.06 + i * 0.4) * 0.4 + 0.6,
            transform: `rotate(${Math.sin(frame * 0.05 + i) * 20}deg) scale(${0.8 + Math.sin(frame * 0.04 + i) * 0.3})`,
          }}
        >
          {note}
        </div>
      ))}
      
      {/* 节拍波纹 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`beat-ripple-${i}`}
          style={{
            position: "absolute",
            width: `${width * (0.037 + i * 0.019)}px`,
            height: `${height * (0.021 + i * 0.010)}px`,
            border: `2px solid ${BEAT_COLORS[i % BEAT_COLORS.length]}30`,
            borderRadius: "50%",
            left: `${60 + i * 10}%`,
            top: `${50 + Math.sin(frame * 0.1 + i * 0.6) * 10}%`,
            transform: `translate(-50%, -50%) scale(${0.5 + Math.abs(Math.sin(frame * 0.15 + i)) * 0.8})`,
            opacity: Math.abs(Math.cos(frame * 0.12 + i)) * 0.5 + 0.3,
          }}
        />
      ))}
      
      <style>{`
        @keyframes beat-note {
          0%, 100% { transform: translateX(-50%) rotate(0deg) scale(1); }
          25% { transform: translateX(-50%) rotate(-5deg) scale(1.1); }
          50% { transform: translateX(-50%) rotate(0deg) scale(0.9); }
          75% { transform: translateX(-50%) rotate(5deg) scale(1.1); }
        }
      `}</style>
    </AbsoluteFill>
  );
};
// 烟雾迷雾样式 - 神秘烟雾迷雾效果
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

export const SmokeMistStyle: React.FC<{
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
      {/* 烟雾背景层 */}
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={`smoke-layer-${i}`}
          style={{
            position: "absolute",
            width: `${100 + i * 50}px`,
            height: `${60 + i * 20}px`,
            background: `radial-gradient(ellipse, 
              rgba(128, 128, 128, ${0.1 + i * 0.02}) 0%, 
              rgba(169, 169, 169, ${0.08 + i * 0.01}) 30%, 
              transparent 70%)`,
            borderRadius: "50%",
            left: `${10 + i * 15 + Math.sin(frame * 0.01 + i * 0.5) * 20}%`,
            top: `${20 + i * 10 + Math.cos(frame * 0.008 + i * 0.3) * 15}%`,
            filter: "blur(20px)",
            opacity: Math.sin(frame * 0.005 + i * 0.4) * 0.3 + 0.4,
            transform: `rotate(${Math.sin(frame * 0.003 + i) * 30}deg) scale(${0.8 + Math.sin(frame * 0.004 + i * 0.2) * 0.4})`,
          }}
        />
      ))}
      
      <div
        style={{
          fontSize,
          color: "transparent",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.95, 1])),
            translateY(interpolate(enterProgress, [0, 1], [18, 0])),
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

            // 烟雾飘动效果
            const smokePhase = (frame * 0.02 + index * 0.4) % (Math.PI * 2);
            const smokeDrift = Math.sin(smokePhase) * 2;
            const smokeOpacity = Math.abs(Math.cos(smokePhase * 0.7)) * 0.3 + 0.7;
            
            // 烟雾渐变色
            const smokeGradient = `linear-gradient(135deg,
              rgba(105, 105, 105, 0.9) 0%,
              rgba(169, 169, 169, 0.8) 25%,
              rgba(211, 211, 211, 0.9) 50%,
              rgba(192, 192, 192, 0.8) 75%,
              rgba(128, 128, 128, 0.9) 100%)`;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : smokeOpacity,
                  
                  // 烟雾文字效果
                  background: isCurrentlyReading 
                    ? smokeGradient
                    : hasBeenRead
                      ? `linear-gradient(45deg, 
                          rgba(169, 169, 169, 0.8) 0%, 
                          rgba(211, 211, 211, 0.7) 100%)`
                      : `linear-gradient(45deg, 
                          rgba(255, 255, 255, 0.9) 0%, 
                          rgba(248, 248, 255, 0.8) 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  
                  // 烟雾背景
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-10px",
                    left: "-15px",
                    right: "-15px",
                    bottom: "-10px",
                    background: `radial-gradient(ellipse at center,
                      rgba(128, 128, 128, 0.2) 0%,
                      rgba(169, 169, 169, 0.15) 40%,
                      rgba(211, 211, 211, 0.1) 70%,
                      transparent 100%)`,
                    borderRadius: "20px",
                    filter: "blur(8px)",
                    zIndex: -1,
                    animation: "smoke-swirl 4s ease-in-out infinite",
                  } : undefined,
                  
                  borderRadius: isCurrentlyReading ? "18px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 烟雾边框
                  border: isCurrentlyReading 
                    ? "2px solid rgba(169, 169, 169, 0.4)" 
                    : "none",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.06) translateY(${-6 + smokeDrift}px) translateX(${smokeDrift * 0.5}px)`
                    : notYetRead 
                      ? "scale(0.94) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 烟雾阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 6px 20px rgba(128, 128, 128, 0.4),
                       inset 0 2px 4px rgba(255, 255, 255, 0.3),
                       inset 0 -2px 4px rgba(105, 105, 105, 0.2),
                       0 0 30px rgba(169, 169, 169, 0.3)`
                    : hasBeenRead
                      ? "0 3px 10px rgba(169, 169, 169, 0.3)"
                      : "0 2px 4px rgba(0, 0, 0, 0.2)",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 2px 8px rgba(105, 105, 105, 0.6),
                       0 0 15px rgba(169, 169, 169, 0.5)`
                    : hasBeenRead
                      ? "0 2px 6px rgba(169, 169, 169, 0.4)"
                      : "0 2px 4px rgba(0, 0, 0, 0.3)",
                  
                  transition: "all 0.5s ease-out",
                  
                  // 烟雾粒子装饰
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-8px",
                    right: "10%",
                    width: "3px",
                    height: "8px",
                    background: "rgba(169, 169, 169, 0.6)",
                    borderRadius: "50%",
                    filter: "blur(1px)",
                    animation: "smoke-rise 3s ease-in-out infinite",
                  } : undefined,
                }}
              >
                {/* 烟雾漩涡 */}
                {isCurrentlyReading && [1, 2].map((i) => (
                  <div
                    key={`smoke-swirl-${i}`}
                    style={{
                      position: "absolute",
                      top: `${-5 - i * 5}px`,
                      left: `${20 + i * 30}%`,
                      width: `${4 + i * 2}px`,
                      height: `${8 + i * 4}px`,
                      background: `rgba(169, 169, 169, ${0.4 - i * 0.1})`,
                      borderRadius: "50%",
                      filter: "blur(2px)",
                      animation: `smoke-particle-${i} ${2 + i}s ease-in-out infinite`,
                    }}
                  />
                ))}
                
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 上升烟雾 */}
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div
          key={`rising-smoke-${i}`}
          style={{
            position: "absolute",
            width: `${2 + i}px`,
            height: `${4 + i * 2}px`,
            background: `rgba(169, 169, 169, ${0.3 - i * 0.03})`,
            borderRadius: "50%",
            left: `${15 + i * 12}%`,
            bottom: `${200 + Math.sin(frame * 0.02 + i * 0.8) * 40}px`,
            opacity: Math.sin(frame * 0.015 + i * 0.5) * 0.4 + 0.6,
            transform: `translateX(${Math.sin(frame * 0.01 + i) * 15}px) scale(${0.5 + Math.sin(frame * 0.008 + i) * 0.3})`,
            filter: "blur(3px)",
          }}
        />
      ))}
      
      {/* 烟雾云团 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`smoke-cloud-${i}`}
          style={{
            position: "absolute",
            width: `${30 + i * 20}px`,
            height: `${15 + i * 10}px`,
            background: `radial-gradient(ellipse, 
              rgba(128, 128, 128, ${0.2 - i * 0.04}) 0%, 
              rgba(169, 169, 169, ${0.15 - i * 0.03}) 50%, 
              transparent 100%)`,
            borderRadius: "50%",
            left: `${25 + i * 25}%`,
            top: `${35 + Math.sin(frame * 0.006 + i * 0.7) * 20}%`,
            opacity: Math.sin(frame * 0.004 + i * 0.3) * 0.3 + 0.5,
            transform: `rotate(${Math.sin(frame * 0.002 + i) * 45}deg) scale(${0.8 + Math.sin(frame * 0.003 + i * 0.4) * 0.4})`,
            filter: "blur(15px)",
          }}
        />
      ))}
      
      <style>{`
        @keyframes smoke-swirl {
          0%, 100% { 
            transform: rotate(0deg) scale(1);
            opacity: 0.2;
          }
          25% { 
            transform: rotate(90deg) scale(1.1);
            opacity: 0.3;
          }
          50% { 
            transform: rotate(180deg) scale(0.9);
            opacity: 0.25;
          }
          75% { 
            transform: rotate(270deg) scale(1.05);
            opacity: 0.35;
          }
        }
        
        @keyframes smoke-rise {
          0% { 
            transform: translateY(0px) scale(1);
            opacity: 0.6;
          }
          100% { 
            transform: translateY(-20px) scale(0.3);
            opacity: 0;
          }
        }
        
        @keyframes smoke-particle-1 {
          0%, 100% { 
            transform: translateY(0px) translateX(0px) scale(1);
            opacity: 0.4;
          }
          50% { 
            transform: translateY(-15px) translateX(8px) scale(0.6);
            opacity: 0.2;
          }
        }
        
        @keyframes smoke-particle-2 {
          0%, 100% { 
            transform: translateY(0px) translateX(0px) scale(1);
            opacity: 0.3;
          }
          50% { 
            transform: translateY(-25px) translateX(-12px) scale(0.4);
            opacity: 0.1;
          }
        }
      `}</style>
    </AbsoluteFill>
  );
};
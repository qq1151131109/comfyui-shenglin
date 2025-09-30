// 火焰燃烧样式 - 真实火焰燃烧效果
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





export const FireFlameStyle: React.FC<{
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

            // 火焰颜色渐变动画
            const flamePhase = (frame * 0.2 + index * 0.5) % (Math.PI * 2);
            const flameIntensity = Math.abs(Math.sin(flamePhase)) * 0.5 + 0.5;
            
            const fireGradient = `linear-gradient(0deg,
              #FF4500 0%,
              #FF6347 ${20 + flameIntensity * 10}%,
              #FF7F50 ${40 + flameIntensity * 15}%,
              #FFA500 ${60 + flameIntensity * 10}%,
              #FFD700 ${80 + flameIntensity * 20}%,
              #FFFF00 100%)`;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 火焰文字渐变
                  background: isCurrentlyReading 
                    ? fireGradient
                    : hasBeenRead
                      ? `linear-gradient(45deg, #FF6347 0%, #FFA500 100%)`
                      : `linear-gradient(45deg, #FFFFFF 0%, #FFF8DC 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundSize: isCurrentlyReading ? "100% 200%" : "100% 100%",
                  
                  // 火焰背景效果
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-10px",
                    left: "-8px",
                    right: "-8px",
                    bottom: "-5px",
                    background: `radial-gradient(ellipse at bottom,
                      transparent 0%,
                      rgba(255, 69, 0, 0.3) 20%,
                      rgba(255, 140, 0, 0.2) 50%,
                      rgba(255, 215, 0, 0.1) 80%,
                      transparent 100%)`,
                    borderRadius: "15px 15px 50% 50%",
                    filter: "blur(3px)",
                    zIndex: -1,
                  } : undefined,
                  
                  borderRadius: isCurrentlyReading ? "12px" : "0px",
                  padding: isCurrentlyReading ? "6px 14px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  // 火焰动画
                  animation: isCurrentlyReading 
                    ? `fire-flicker ${2 + random(`duration-${t.fromMs}`) * 2}s ease-in-out infinite` 
                    : "none",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.1) translateY(${-6 + Math.sin(flamePhase) * 2}px)`
                    : notYetRead 
                      ? "scale(0.9) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  // 火焰发光
                  filter: isCurrentlyReading 
                    ? `drop-shadow(0 0 10px #FF4500) drop-shadow(0 0 20px #FF6347) drop-shadow(0 0 30px #FFA500)`
                    : hasBeenRead
                      ? "drop-shadow(0 0 8px #FF6347)"
                      : "drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 0 10px #FF4500,
                       0 0 20px #FF6347,
                       0 0 30px #FFA500,
                       2px 2px 4px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? "0 0 8px #FF6347"
                      : "2px 2px 4px rgba(0, 0, 0, 0.5)",
                  
                  transition: "all 0.3s ease-out",
                }}
              >
                {/* 火焰粒子 */}
                {isCurrentlyReading && [1, 2, 3, 4].map((i) => (
                  <div
                    key={`flame-particle-${i}`}
                    style={{
                      position: "absolute",
                      width: `${1 + i}px`,
                      height: `${height * (0.001 + i * 0.001)}px`,
                      background: i % 2 === 0 ? "#FF4500" : "#FFD700",
                      borderRadius: "50% 50% 50% 50% / 60% 60% 40% 40%",
                      left: `${random(`particle-x-${t.fromMs}-${i}-${Math.floor(frame / 5)}`) * 100}%`,
                      top: `${-10 - random(`particle-y-${t.fromMs}-${i}-${Math.floor(frame / 3)}`) * 20}px`,
                      opacity: random(`particle-op-${t.fromMs}-${i}-${Math.floor(frame / 4)}`) * 0.8 + 0.2,
                      transform: `scale(${0.5 + random(`particle-scale-${t.fromMs}-${i}-${Math.floor(frame / 6)}`) * 1}) 
                                 rotate(${random(`particle-rot-${t.fromMs}-${i}-${frame}`) * 360}deg)`,
                      filter: "blur(0.5px)",
                    }}
                  />
                ))}
                
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 火焰环境粒子 */}
      {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
        <div
          key={`fire-ember-${i}`}
          style={{
            position: "absolute",
            width: `${1 + random(`ember-size-${i}`) * 3}px`,
            height: `${2 + random(`ember-size-${i}`) * 4}px`,
            background: i % 3 === 0 ? "#FF4500" : i % 3 === 1 ? "#FFD700" : "#FF6347",
            borderRadius: "50%",
            left: `${10 + i * 10 + random(`ember-x-${i}-${Math.floor(frame / 10)}`) * 20}%`,
            top: `${60 + random(`ember-y-${i}-${Math.floor(frame / 8)}`) * 40}%`,
            opacity: random(`ember-op-${i}-${Math.floor(frame / 12)}`) * 0.7 + 0.3,
            transform: `translateY(${-random(`ember-float-${i}-${Math.floor(frame / 6)}`) * 30}px) 
                       scale(${0.3 + random(`ember-scale-${i}-${Math.floor(frame / 8)}`) * 0.7})`,
            filter: "blur(1px)",
          }}
        />
      ))}
      
      {/* 热浪效果 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`heat-wave-${i}`}
          style={{
            position: "absolute",
            width: "100%",
            height: `${2 + i}px`,
            background: `linear-gradient(90deg, 
              transparent, 
              rgba(255, 69, 0, ${0.1 + i * 0.05}), 
              transparent)`,
            top: `${50 + i * 10 + Math.sin(frame * 0.1 + i) * 5}%`,
            opacity: Math.sin(frame * 0.08 + i * 0.5) * 0.3 + 0.4,
            filter: "blur(2px)",
            transform: `scaleX(${0.8 + Math.sin(frame * 0.05 + i) * 0.2})`,
          }}
        />
      ))}
      
      <style>{`
        @keyframes fire-flicker {
          0%, 100% { 
            background-size: 100% 200%;
            background-position: 0% 0%;
            filter: drop-shadow(0 0 10px #FF4500) drop-shadow(0 0 20px #FF6347);
          }
          25% { 
            background-size: 120% 220%;
            background-position: 0% 20%;
            filter: drop-shadow(0 0 15px #FF4500) drop-shadow(0 0 25px #FF6347) drop-shadow(0 0 35px #FFA500);
          }
          50% { 
            background-size: 110% 210%;
            background-position: 0% 40%;
            filter: drop-shadow(0 0 12px #FF4500) drop-shadow(0 0 22px #FF6347);
          }
          75% { 
            background-size: 130% 230%;
            background-position: 0% 60%;
            filter: drop-shadow(0 0 18px #FF4500) drop-shadow(0 0 28px #FF6347) drop-shadow(0 0 38px #FFA500);
          }
        }
      `}</style>
    </AbsoluteFill>
  );
};
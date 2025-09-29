// 星空粒子样式 - 梦幻星空粒子效果
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

const container: React.CSSProperties = {
  justifyContent: "center",
  alignItems: "center",
  top: undefined,
  bottom: 350,
  height: 150,
};

const DESIRED_FONT_SIZE = 120;
const STAR_COLORS = ["#FFD700", "#FFFFFF", "#FFF8DC", "#F0E68C", "#FFFFE0", "#FFFACD"];

export const StarryParticleStyle: React.FC<{
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

            // 星光闪烁效果
            const starPhase = (frame * 0.08 + index * 0.6) % (Math.PI * 2);
            const sparkleIntensity = Math.abs(Math.sin(starPhase)) * 0.5 + 0.5;
            const shouldSparkle = isCurrentlyReading && frame % 8 < 2;
            
            // 星空渐变
            const starGradient = `linear-gradient(45deg,
              #FFD700 0%,
              #FFFFFF ${20 + sparkleIntensity * 30}%,
              #FFF8DC ${40 + sparkleIntensity * 20}%,
              #F0E68C ${60 + sparkleIntensity * 30}%,
              #FFFFFF ${80 + sparkleIntensity * 20}%,
              #FFD700 100%)`;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 星光文字渐变
                  background: isCurrentlyReading 
                    ? (shouldSparkle ? "#FFFFFF" : starGradient)
                    : hasBeenRead
                      ? `linear-gradient(45deg, #FFD700 0%, #FFF8DC 100%)`
                      : `linear-gradient(45deg, #FFFFFF 0%, #F5F5DC 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundSize: isCurrentlyReading ? "300% 300%" : "100% 100%",
                  
                  // 星光背景
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "-8px",
                    left: "-12px",
                    right: "-12px",
                    bottom: "-8px",
                    background: `radial-gradient(ellipse at center,
                      rgba(255, 215, 0, 0.3) 0%,
                      rgba(255, 248, 220, 0.2) 40%,
                      rgba(240, 230, 140, 0.1) 70%,
                      transparent 100%)`,
                    borderRadius: "20px",
                    filter: "blur(6px)",
                    zIndex: -1,
                    animation: "star-twinkle 2s ease-in-out infinite",
                  } : undefined,
                  
                  borderRadius: isCurrentlyReading ? "15px" : "0px",
                  padding: isCurrentlyReading ? "8px 16px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 星光边框
                  border: isCurrentlyReading 
                    ? `2px solid ${shouldSparkle ? '#FFFFFF' : 'rgba(255, 215, 0, 0.6)'}` 
                    : "none",
                  
                  transform: isCurrentlyReading 
                    ? `scale(${1.08 + (shouldSparkle ? 0.02 : 0)}) translateY(${-6 + Math.sin(starPhase) * 1}px)`
                    : notYetRead 
                      ? "scale(0.92) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 星光发光
                  boxShadow: isCurrentlyReading 
                    ? `0 0 15px ${shouldSparkle ? '#FFFFFF' : '#FFD700'},
                       0 0 25px ${shouldSparkle ? '#FFFFFF' : '#FFD700'},
                       0 0 35px ${shouldSparkle ? '#FFFFFF' : '#FFF8DC'},
                       inset 0 0 15px rgba(255, 215, 0, 0.2)`
                    : hasBeenRead
                      ? "0 0 10px #FFD70080"
                      : "0 2px 4px rgba(0, 0, 0, 0.2)",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 0 8px ${shouldSparkle ? '#FFFFFF' : '#FFD700'},
                       0 0 15px ${shouldSparkle ? '#FFFFFF' : '#FFD700'},
                       0 0 25px ${shouldSparkle ? '#FFFFFF' : '#FFF8DC'},
                       2px 2px 4px rgba(0, 0, 0, 0.6)`
                    : hasBeenRead
                      ? "0 0 8px #FFD700"
                      : "2px 2px 4px rgba(0, 0, 0, 0.4)",
                  
                  transition: shouldSparkle ? "none" : "all 0.3s ease-out",
                  
                  // 星光动画
                  animation: isCurrentlyReading 
                    ? "star-pulse 1.5s ease-in-out infinite" 
                    : "none",
                }}
              >
                {/* 星光粒子 */}
                {isCurrentlyReading && [1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={`star-particle-${i}`}
                    style={{
                      position: "absolute",
                      width: "2px",
                      height: "2px",
                      background: STAR_COLORS[i % STAR_COLORS.length],
                      left: `${random(`star-x-${t.fromMs}-${i}-${Math.floor(frame / 6)}`) * 120 - 10}%`,
                      top: `${random(`star-y-${t.fromMs}-${i}-${Math.floor(frame / 8)}`) * 120 - 10}%`,
                      opacity: random(`star-op-${t.fromMs}-${i}-${Math.floor(frame / 4)}`) * 0.8 + 0.2,
                      transform: `scale(${0.5 + random(`star-scale-${t.fromMs}-${i}-${Math.floor(frame / 10)}`)} * 2)`,
                      boxShadow: `0 0 6px ${STAR_COLORS[i % STAR_COLORS.length]}`,
                      borderRadius: "50%",
                      animation: `star-float-${i % 3} ${2 + i * 0.5}s ease-in-out infinite`,
                    }}
                  />
                ))}
                
                {/* 星座装饰 */}
                {isCurrentlyReading && (
                  <div
                    style={{
                      position: "absolute",
                      top: "-20px",
                      left: "50%",
                      transform: "translateX(-50%)",
                      fontSize: "14px",
                      color: shouldSparkle ? "#FFFFFF" : "#FFD700",
                      opacity: 0.8,
                      textShadow: `0 0 8px ${shouldSparkle ? '#FFFFFF' : '#FFD700'}`,
                    }}
                  >
                    ✨
                  </div>
                )}
                
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 背景星星 */}
      {[...Array(20)].map((_, i) => (
        <div
          key={`bg-star-${i}`}
          style={{
            position: "absolute",
            width: `${1 + random(`bg-star-size-${i}`) * 3}px`,
            height: `${1 + random(`bg-star-size-${i}`) * 3}px`,
            background: STAR_COLORS[i % STAR_COLORS.length],
            left: `${random(`bg-star-x-${i}`) * 100}%`,
            top: `${random(`bg-star-y-${i}`) * 100}%`,
            opacity: random(`bg-star-op-${i}-${Math.floor(frame / 15)}`) * 0.8 + 0.2,
            transform: `scale(${0.3 + random(`bg-star-scale-${i}-${Math.floor(frame / 12)}`)} * 1.5)`,
            borderRadius: "50%",
            boxShadow: `0 0 4px ${STAR_COLORS[i % STAR_COLORS.length]}`,
            animation: `bg-star-twinkle-${i % 4} ${3 + random(`bg-star-duration-${i}`) * 4}s ease-in-out infinite`,
          }}
        />
      ))}
      
      {/* 流星 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`meteor-${i}`}
          style={{
            position: "absolute",
            width: `${15 + i * 10}px`,
            height: "2px",
            background: `linear-gradient(90deg, 
              transparent, 
              ${STAR_COLORS[i % STAR_COLORS.length]}, 
              transparent)`,
            left: `${-20 + (frame * (0.5 + i * 0.3)) % 140}%`,
            top: `${20 + i * 20 + Math.sin(frame * 0.01 + i) * 10}%`,
            opacity: Math.sin((frame * 0.02 + i) % (Math.PI * 2)) * 0.6 + 0.4,
            transform: `rotate(${-15 + i * 5}deg)`,
            filter: "blur(1px)",
            boxShadow: `0 0 8px ${STAR_COLORS[i % STAR_COLORS.length]}`,
          }}
        />
      ))}
      
      {/* 星云效果 */}
      {[1, 2, 3].map((i) => (
        <div
          key={`nebula-${i}`}
          style={{
            position: "absolute",
            width: `${80 + i * 40}px`,
            height: `${40 + i * 20}px`,
            background: `radial-gradient(ellipse, 
              rgba(255, 215, 0, ${0.05 + i * 0.02}) 0%, 
              rgba(255, 248, 220, ${0.03 + i * 0.01}) 50%, 
              transparent 100%)`,
            borderRadius: "50%",
            left: `${20 + i * 20 + Math.sin(frame * 0.005 + i * 0.7) * 15}%`,
            top: `${30 + i * 15 + Math.cos(frame * 0.003 + i * 0.5) * 10}%`,
            opacity: Math.sin(frame * 0.002 + i * 0.3) * 0.3 + 0.4,
            transform: `rotate(${Math.sin(frame * 0.001 + i) * 60}deg) scale(${0.8 + Math.sin(frame * 0.002 + i * 0.4) * 0.4})`,
            filter: "blur(25px)",
          }}
        />
      ))}
      
      <style>{`
        @keyframes star-twinkle {
          0%, 100% { 
            opacity: 0.3;
            transform: scale(1);
          }
          50% { 
            opacity: 0.6;
            transform: scale(1.1);
          }
        }
        
        @keyframes star-pulse {
          0%, 100% { 
            background-size: 300% 300%;
            background-position: 0% 50%;
          }
          50% { 
            background-size: 400% 400%;
            background-position: 100% 50%;
          }
        }
        
        @keyframes star-float-0 {
          0%, 100% { transform: translateY(0px) translateX(0px) scale(1); }
          50% { transform: translateY(-8px) translateX(4px) scale(1.2); }
        }
        
        @keyframes star-float-1 {
          0%, 100% { transform: translateY(0px) translateX(0px) scale(1); }
          50% { transform: translateY(-6px) translateX(-6px) scale(1.1); }
        }
        
        @keyframes star-float-2 {
          0%, 100% { transform: translateY(0px) translateX(0px) scale(1); }
          50% { transform: translateY(-10px) translateX(2px) scale(1.3); }
        }
        
        @keyframes bg-star-twinkle-0 {
          0%, 100% { opacity: 0.2; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.2); }
        }
        
        @keyframes bg-star-twinkle-1 {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 0.9; transform: scale(1.1); }
        }
        
        @keyframes bg-star-twinkle-2 {
          0%, 100% { opacity: 0.4; transform: scale(1.1); }
          50% { opacity: 0.7; transform: scale(0.9); }
        }
        
        @keyframes bg-star-twinkle-3 {
          0%, 100% { opacity: 0.1; transform: scale(1.3); }
          50% { opacity: 1; transform: scale(0.7); }
        }
      `}</style>
    </AbsoluteFill>
  );
};
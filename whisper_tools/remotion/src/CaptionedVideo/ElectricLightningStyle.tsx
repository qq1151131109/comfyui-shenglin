// 电流闪电样式 - 强烈电流闪电效果
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
const ELECTRIC_COLORS = ["#00FFFF", "#FFFFFF", "#E6E6FA", "#FFFF00", "#00BFFF"];

export const ElectricLightningStyle: React.FC<{
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
            translateY(interpolate(enterProgress, [0, 1], [25, 0])),
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

            // 电流闪烁效果
            const electricPulse = Math.abs(Math.sin(frame * 0.5 + index * 0.3));
            const shouldFlash = isCurrentlyReading && frame % 6 < 2;
            const shouldSpark = isCurrentlyReading && frame % 12 < 1;
            
            // 电流渐变
            const electricGradient = `linear-gradient(45deg,
              #00FFFF 0%,
              #FFFFFF ${20 + electricPulse * 30}%,
              #00BFFF ${40 + electricPulse * 20}%,
              #FFFF00 ${60 + electricPulse * 30}%,
              #FFFFFF ${80 + electricPulse * 20}%,
              #00FFFF 100%)`;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  
                  // 电流文字渐变
                  background: isCurrentlyReading 
                    ? (shouldFlash ? "#FFFFFF" : electricGradient)
                    : hasBeenRead
                      ? `linear-gradient(45deg, #00FFFF 0%, #00BFFF 100%)`
                      : `linear-gradient(45deg, #FFFFFF 0%, #E6E6FA 100%)`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundSize: isCurrentlyReading ? "300% 300%" : "100% 100%",
                  
                  // 电流背景
                  backgroundColor: isCurrentlyReading 
                    ? (shouldFlash ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 255, 255, 0.1)")
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "8px" : "0px",
                  padding: isCurrentlyReading ? "6px 14px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  // 电流边框
                  border: isCurrentlyReading 
                    ? `2px solid ${shouldFlash ? '#FFFFFF' : '#00FFFF'}` 
                    : "none",
                  
                  // 电流动画
                  animation: isCurrentlyReading 
                    ? `electric-pulse 0.1s ease-in-out infinite, electric-glow 2s ease-in-out infinite` 
                    : "none",
                  
                  transform: isCurrentlyReading 
                    ? `scale(${1.1 + (shouldFlash ? 0.05 : 0)}) 
                       translateY(${-6 + (shouldSpark ? random(`spark-y-${t.fromMs}-${frame}`) * 4 - 2 : 0)}px)
                       translateX(${shouldSpark ? random(`spark-x-${t.fromMs}-${frame}`) * 2 - 1 : 0}px)`
                    : notYetRead 
                      ? "scale(0.9) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  // 电流发光
                  boxShadow: isCurrentlyReading 
                    ? `0 0 10px ${shouldFlash ? '#FFFFFF' : '#00FFFF'},
                       0 0 20px ${shouldFlash ? '#FFFFFF' : '#00FFFF'},
                       0 0 30px ${shouldFlash ? '#FFFFFF' : '#00BFFF'},
                       ${shouldFlash ? '0 0 40px #FFFFFF,' : ''}
                       inset 0 0 10px rgba(0, 255, 255, 0.3)`
                    : hasBeenRead
                      ? "0 0 8px #00FFFF80"
                      : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `0 0 5px ${shouldFlash ? '#FFFFFF' : '#00FFFF'},
                       0 0 10px ${shouldFlash ? '#FFFFFF' : '#00FFFF'},
                       0 0 15px ${shouldFlash ? '#FFFFFF' : '#00BFFF'},
                       ${shouldFlash ? '0 0 20px #FFFFFF,' : ''}
                       2px 2px 4px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? "0 0 8px #00FFFF"
                      : "2px 2px 4px rgba(0, 0, 0, 0.5)",
                  
                  // 电流轮廓
                  outline: isCurrentlyReading 
                    ? `1px solid ${shouldFlash ? '#FFFFFF80' : '#00FFFF80'}` 
                    : "none",
                  outlineOffset: shouldSpark ? `${random(`outline-${t.fromMs}-${frame}`) * 3}px` : "1px",
                  
                  transition: shouldFlash || shouldSpark ? "none" : "all 0.2s ease-out",
                }}
              >
                {/* 电火花效果 */}
                {isCurrentlyReading && shouldSpark && [1, 2, 3].map((i) => (
                  <div
                    key={`spark-${i}`}
                    style={{
                      position: "absolute",
                      width: "2px",
                      height: `${2 + i}px`,
                      background: ELECTRIC_COLORS[i % ELECTRIC_COLORS.length],
                      left: `${random(`spark-pos-x-${t.fromMs}-${i}-${frame}`) * 100}%`,
                      top: `${random(`spark-pos-y-${t.fromMs}-${i}-${frame}`) * 100}%`,
                      transform: `rotate(${random(`spark-rot-${t.fromMs}-${i}-${frame}`) * 360}deg)`,
                      opacity: random(`spark-op-${t.fromMs}-${i}-${frame}`) * 0.8 + 0.2,
                      filter: "blur(0.5px)",
                      boxShadow: `0 0 4px ${ELECTRIC_COLORS[i % ELECTRIC_COLORS.length]}`,
                    }}
                  />
                ))}
                
                {/* 闪电装饰 */}
                {isCurrentlyReading && (
                  <div
                    style={{
                      position: "absolute",
                      top: "-15px",
                      left: "50%",
                      transform: "translateX(-50%)",
                      fontSize: "12px",
                      color: shouldFlash ? "#FFFFFF" : "#00FFFF",
                      opacity: shouldSpark ? 1 : 0.7,
                    }}
                  >
                    ⚡
                  </div>
                )}
                
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 背景闪电 */}
      {[1, 2, 3, 4].map((i) => (
        <div
          key={`bg-lightning-${i}`}
          style={{
            position: "absolute",
            width: "2px",
            height: `${20 + i * 15}px`,
            background: `linear-gradient(to bottom, 
              transparent, 
              ${ELECTRIC_COLORS[i % ELECTRIC_COLORS.length]}, 
              transparent)`,
            left: `${15 + i * 20}%`,
            top: `${10 + random(`bg-lightning-${i}-${Math.floor(frame / 5)}`) * 60}%`,
            opacity: random(`bg-lightning-op-${i}-${Math.floor(frame / 8)}`) * 0.6 + 0.2,
            transform: `rotate(${random(`bg-lightning-rot-${i}-${Math.floor(frame / 6)}`) * 30 - 15}deg) 
                       scaleY(${0.5 + random(`bg-lightning-scale-${i}-${Math.floor(frame / 4)}`) * 1.5})`,
            filter: "blur(1px)",
            boxShadow: `0 0 8px ${ELECTRIC_COLORS[i % ELECTRIC_COLORS.length]}`,
          }}
        />
      ))}
      
      {/* 电流粒子 */}
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div
          key={`electric-particle-${i}`}
          style={{
            position: "absolute",
            width: "1px",
            height: "1px",
            background: "#FFFFFF",
            left: `${20 + i * 12 + random(`particle-x-${i}-${Math.floor(frame / 3)}`) * 20}%`,
            top: `${30 + random(`particle-y-${i}-${Math.floor(frame / 4)}`) * 40}%`,
            opacity: random(`particle-op-${i}-${Math.floor(frame / 2)}`) * 0.8 + 0.2,
            boxShadow: `0 0 6px #00FFFF, 0 0 10px #FFFFFF`,
            transform: `scale(${1 + random(`particle-scale-${i}-${Math.floor(frame / 5)}`)} * 3)`,
            filter: "blur(0.5px)",
          }}
        />
      ))}
      
      <style>{`
        @keyframes electric-pulse {
          0%, 100% { 
            filter: brightness(1) contrast(1);
          }
          50% { 
            filter: brightness(1.3) contrast(1.2);
          }
        }
        
        @keyframes electric-glow {
          0%, 100% { 
            background-position: 0% 50%;
          }
          25% { 
            background-position: 100% 0%;
          }
          50% { 
            background-position: 200% 50%;
          }
          75% { 
            background-position: 300% 100%;
          }
        }
      `}</style>
    </AbsoluteFill>
  );
};
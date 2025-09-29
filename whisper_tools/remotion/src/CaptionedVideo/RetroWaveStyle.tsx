// 复古波浪样式 - 80年代风格的字幕效果
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TheBoldFont } from "../load-font";
import { fitText } from "@remotion/layout-utils";
import { makeTransform, scale, translateY, translateX } from "@remotion/animation-utils";
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
const RETRO_PINK = "#FF0080";
const RETRO_CYAN = "#00FFFF";
const RETRO_PURPLE = "#8000FF";

export const RetroWaveStyle: React.FC<{
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
      {/* 背景渐变层 */}
      <div
        style={{
          position: "absolute",
          width: "120%",
          height: "200%",
          background: `linear-gradient(45deg, 
            rgba(128, 0, 255, 0.1) 0%, 
            rgba(255, 0, 128, 0.1) 25%, 
            rgba(0, 255, 255, 0.1) 50%, 
            rgba(255, 0, 128, 0.1) 75%, 
            rgba(128, 0, 255, 0.1) 100%)`,
          transform: `translateX(${Math.sin(frame * 0.02) * 20}px)`,
          opacity: 0.3,
        }}
      />
      
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.8, 1])),
            translateY(interpolate(enterProgress, [0, 1], [40, 0])),
          ]),
          // 复古3D效果
          filter: "drop-shadow(3px 3px 0px #FF0080) drop-shadow(6px 6px 0px #8000FF)",
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

            // 波浪动画偏移
            const waveOffset = Math.sin((frame * 0.1) + (index * 0.5)) * 8;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? RETRO_PINK
                    : hasBeenRead 
                      ? RETRO_CYAN
                      : "white",
                  
                  // 复古渐变文字
                  background: isCurrentlyReading 
                    ? `linear-gradient(45deg, ${RETRO_PINK} 0%, ${RETRO_CYAN} 50%, ${RETRO_PURPLE} 100%)`
                    : hasBeenRead
                      ? `linear-gradient(45deg, ${RETRO_CYAN} 0%, rgba(255, 255, 255, 0.8) 100%)`
                      : "none",
                  WebkitBackgroundClip: isCurrentlyReading || hasBeenRead ? "text" : "initial",
                  WebkitTextFillColor: isCurrentlyReading || hasBeenRead ? "transparent" : "white",
                  
                  // 多层阴影效果
                  textShadow: isCurrentlyReading 
                    ? `2px 2px 0px ${RETRO_PINK}, 4px 4px 0px ${RETRO_PURPLE}, 6px 6px 0px ${RETRO_CYAN}, 8px 8px 20px rgba(0, 0, 0, 0.8)`
                    : hasBeenRead
                      ? `1px 1px 0px ${RETRO_CYAN}, 2px 2px 0px ${RETRO_PURPLE}, 4px 4px 10px rgba(0, 0, 0, 0.6)`
                      : "2px 2px 0px rgba(255, 255, 255, 0.3), 4px 4px 8px rgba(0, 0, 0, 0.7)",
                  
                  // 背景框效果
                  backgroundColor: isCurrentlyReading 
                    ? "rgba(255, 0, 128, 0.2)" 
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "8px" : "0px",
                  padding: isCurrentlyReading ? "4px 12px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  border: isCurrentlyReading ? `2px solid ${RETRO_CYAN}` : "none",
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-2px" : "0px",
                  
                  transition: "all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55)",
                  
                  // 波浪式变换
                  transform: isCurrentlyReading 
                    ? `scale(1.3) translateY(${-10 + waveOffset}px) rotateZ(${Math.sin(frame * 0.05) * 3}deg) perspective(500px) rotateX(${Math.sin(frame * 0.03) * 5}deg)`
                    : notYetRead 
                      ? `scale(0.7) translateY(${20 + waveOffset}px) rotateZ(-3deg)`
                      : `scale(1) translateY(${waveOffset}px) rotateZ(0deg)`,
                  
                  // 霓虹外发光
                  boxShadow: isCurrentlyReading 
                    ? `0 0 20px ${RETRO_PINK}, 0 0 40px ${RETRO_CYAN}, inset 0 0 20px rgba(255, 255, 255, 0.1)`
                    : hasBeenRead
                      ? `0 0 10px ${RETRO_CYAN}`
                      : "none",
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
    </AbsoluteFill>
  );
};
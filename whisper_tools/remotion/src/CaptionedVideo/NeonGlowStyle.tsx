// 霓虹发光样式 - 科技感十足的字幕效果
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
const NEON_COLOR = "#00FFFF"; // 青色霓虹
const ACCENT_COLOR = "#FF0080"; // 粉色重点色

export const NeonGlowStyle: React.FC<{
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
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [30, 0])),
          ]),
          filter: "drop-shadow(0 0 20px rgba(0, 255, 255, 0.8))",
        }}
      >
        <span>
          {page.tokens.map((t) => {
            const startRelativeToSequence = t.fromMs - page.startMs;
            const endRelativeToSequence = t.toMs - page.startMs;

            const isCurrentlyReading = 
              startRelativeToSequence <= timeInMs &&
              endRelativeToSequence > timeInMs;
            
            const hasBeenRead = endRelativeToSequence <= timeInMs;
            const notYetRead = startRelativeToSequence > timeInMs;

            // 添加闪烁效果
            const pulseIntensity = Math.sin((frame * 0.3) % (Math.PI * 2)) * 0.5 + 0.5;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? ACCENT_COLOR
                    : hasBeenRead 
                      ? "rgba(255, 255, 255, 0.7)"
                      : "white",
                  
                  // 霓虹发光效果
                  textShadow: isCurrentlyReading 
                    ? `0 0 10px ${NEON_COLOR}, 0 0 20px ${NEON_COLOR}, 0 0 30px ${ACCENT_COLOR}, 0 0 40px ${ACCENT_COLOR}`
                    : hasBeenRead
                      ? `0 0 5px ${NEON_COLOR}, 0 0 10px rgba(0, 255, 255, 0.5)`
                      : "0 0 8px rgba(255, 255, 255, 0.8), 0 0 16px rgba(0, 255, 255, 0.3)",
                  
                  // 背景发光
                  backgroundColor: isCurrentlyReading 
                    ? `rgba(255, 0, 128, ${0.2 + pulseIntensity * 0.3})` 
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "15px" : "0px",
                  padding: isCurrentlyReading ? "6px 15px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-3px" : "0px",
                  
                  transition: "all 0.3s ease-out",
                  
                  transform: isCurrentlyReading 
                    ? `scale(${1.2 + pulseIntensity * 0.1}) translateY(-5px) rotateZ(${Math.sin(frame * 0.1) * 2}deg)`
                    : notYetRead 
                      ? "scale(0.7) translateY(15px) rotateZ(-2deg)"
                      : "scale(1) translateY(0px) rotateZ(0deg)",
                  
                  // 边框发光
                  border: isCurrentlyReading 
                    ? `2px solid rgba(0, 255, 255, ${0.5 + pulseIntensity * 0.5})` 
                    : "none",
                  
                  // 外发光
                  boxShadow: isCurrentlyReading 
                    ? `0 0 20px rgba(255, 0, 128, 0.8), inset 0 0 20px rgba(0, 255, 255, 0.3)`
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
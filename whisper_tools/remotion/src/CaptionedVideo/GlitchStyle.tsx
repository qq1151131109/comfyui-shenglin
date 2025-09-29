// 故障风格样式 - 数字故障效果的字幕
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
const GLITCH_RED = "#FF0040";
const GLITCH_CYAN = "#00FFFF";
const GLITCH_GREEN = "#40FF00";

export const GlitchStyle: React.FC<{
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

  // 故障强度随时间变化
  const glitchIntensity = Math.sin(frame * 0.5) * 0.5 + 0.5;
  const shouldGlitch = frame % 20 < 3; // 每20帧故障3帧

  return (
    <AbsoluteFill style={container}>
      {/* 主文字层 */}
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.9, 1])),
            translateY(interpolate(enterProgress, [0, 1], [25, 0])),
            translateX(shouldGlitch ? random(`glitch-${frame}`) * 8 - 4 : 0),
          ]),
          WebkitTextStroke: "2px black",
          paintOrder: "stroke",
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

            // 为每个词汇生成独特的故障效果
            const wordGlitch = shouldGlitch && random(`word-${t.fromMs}-${Math.floor(frame / 5)}`) > 0.7;
            const glitchOffset = wordGlitch ? random(`offset-${t.fromMs}-${frame}`) * 6 - 3 : 0;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? (wordGlitch ? GLITCH_RED : GLITCH_GREEN)
                    : hasBeenRead 
                      ? (wordGlitch ? GLITCH_CYAN : "rgba(255, 255, 255, 0.8)")
                      : "white",
                  
                  // 故障背景
                  backgroundColor: isCurrentlyReading && wordGlitch
                    ? `rgba(255, 0, 64, ${0.3 * glitchIntensity})` 
                    : isCurrentlyReading
                      ? "rgba(64, 255, 0, 0.2)"
                      : "transparent",
                  borderRadius: isCurrentlyReading ? "6px" : "0px",
                  padding: isCurrentlyReading ? "3px 10px" : "0px",
                  margin: isCurrentlyReading ? "0 3px" : "0",
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-1px" : "0px",
                  
                  // 故障边框
                  border: wordGlitch 
                    ? `1px solid ${random(`border-${t.fromMs}-${frame}`) > 0.5 ? GLITCH_RED : GLITCH_CYAN}`
                    : isCurrentlyReading
                      ? `1px solid ${GLITCH_GREEN}`
                      : "none",
                  
                  transition: wordGlitch ? "none" : "all 0.2s ease-out",
                  
                  transform: isCurrentlyReading 
                    ? `scale(${1.15 + (wordGlitch ? glitchIntensity * 0.3 : 0)}) translateY(${-5 + glitchOffset}px) translateX(${glitchOffset}px) ${wordGlitch ? `skewX(${glitchIntensity * 10}deg)` : ''}`
                    : notYetRead 
                      ? "scale(0.85) translateY(12px)"
                      : "scale(1) translateY(0px)",
                  
                  // 故障文字阴影
                  textShadow: wordGlitch 
                    ? `${glitchOffset}px 0 ${GLITCH_RED}, ${-glitchOffset}px 0 ${GLITCH_CYAN}, 0 0 10px rgba(255, 255, 255, 0.8)`
                    : isCurrentlyReading
                      ? `2px 0 ${GLITCH_GREEN}, -1px 0 ${GLITCH_CYAN}, 0 2px 8px rgba(0, 0, 0, 0.8)`
                      : hasBeenRead
                        ? "1px 1px 4px rgba(0, 0, 0, 0.6)"
                        : "2px 2px 6px rgba(0, 0, 0, 0.7)",
                  
                  // 故障滤镜
                  filter: wordGlitch 
                    ? `hue-rotate(${random(`hue-${t.fromMs}-${frame}`) * 360}deg) saturate(${1.5 + glitchIntensity}) contrast(${1.2 + glitchIntensity * 0.5})`
                    : isCurrentlyReading
                      ? "saturate(1.3) contrast(1.1)"
                      : "none",
                  
                  // 外发光
                  boxShadow: wordGlitch 
                    ? `0 0 20px ${GLITCH_RED}, inset 0 0 10px ${GLITCH_CYAN}`
                    : isCurrentlyReading
                      ? `0 0 15px ${GLITCH_GREEN}`
                      : "none",
                }}
              >
                {/* 故障重影效果 */}
                {wordGlitch && isCurrentlyReading && (
                  <>
                    <span
                      style={{
                        position: "absolute",
                        top: 0,
                        left: `${glitchOffset * 2}px`,
                        color: GLITCH_RED,
                        opacity: 0.7,
                        zIndex: -1,
                        mixBlendMode: "screen",
                      }}
                    >
                      {t.text}
                    </span>
                    <span
                      style={{
                        position: "absolute",
                        top: 0,
                        left: `${-glitchOffset}px`,
                        color: GLITCH_CYAN,
                        opacity: 0.7,
                        zIndex: -1,
                        mixBlendMode: "screen",
                      }}
                    >
                      {t.text}
                    </span>
                  </>
                )}
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
    </AbsoluteFill>
  );
};
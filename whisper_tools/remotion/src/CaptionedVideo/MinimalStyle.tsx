// 极简风格样式 - 现代简约的字幕效果
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
const ACCENT_COLOR = "#6C5CE7"; // 优雅的紫色
const SOFT_GRAY = "#74B9FF"; // 柔和的蓝色

export const MinimalStyle: React.FC<{
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
      {/* 背景模糊圆形装饰 */}
      <div
        style={{
          position: "absolute",
          width: "200px",
          height: "60px",
          background: `linear-gradient(90deg, ${ACCENT_COLOR}20, ${SOFT_GRAY}20)`,
          borderRadius: "30px",
          filter: "blur(20px)",
          opacity: 0.6,
          transform: `scale(${interpolate(enterProgress, [0, 1], [0.5, 1])})`,
        }}
      />
      
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.95, 1])),
            translateY(interpolate(enterProgress, [0, 1], [20, 0])),
          ]),
          // 柔和阴影
          filter: "drop-shadow(0 4px 20px rgba(0, 0, 0, 0.3))",
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

            // 柔和的呼吸效果
            const breathe = Math.sin(frame * 0.08) * 0.05 + 1;

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
                      ? SOFT_GRAY
                      : "white",
                  
                  // 极简背景
                  backgroundColor: isCurrentlyReading 
                    ? "rgba(255, 255, 255, 0.95)" 
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "25px" : "0px",
                  padding: isCurrentlyReading ? "8px 20px" : "0px",
                  margin: isCurrentlyReading ? "0 6px" : "0",
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 柔和边框
                  border: isCurrentlyReading 
                    ? `2px solid ${ACCENT_COLOR}40` 
                    : "none",
                  
                  transition: "all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                  
                  // 柔和变换
                  transform: isCurrentlyReading 
                    ? `scale(${1.08 * breathe}) translateY(-8px)`
                    : notYetRead 
                      ? "scale(0.92) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  // 柔和阴影
                  textShadow: isCurrentlyReading 
                    ? "none" // 白色背景上不需要阴影
                    : hasBeenRead
                      ? `0 2px 12px ${SOFT_GRAY}60`
                      : "0 2px 8px rgba(0, 0, 0, 0.4)",
                  
                  // 外部阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 8px 32px ${ACCENT_COLOR}30, 0 4px 16px rgba(0, 0, 0, 0.1)`
                    : "none",
                  
                  // 字体权重
                  fontWeight: isCurrentlyReading ? "900" : "700",
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* 底部装饰线条 */}
      <div
        style={{
          position: "absolute",
          bottom: "-20px",
          left: "50%",
          transform: "translateX(-50%)",
          width: `${interpolate(enterProgress, [0, 1], [0, 100])}px`,
          height: "3px",
          background: `linear-gradient(90deg, transparent, ${ACCENT_COLOR}, transparent)`,
          borderRadius: "2px",
          opacity: 0.6,
        }}
      />
    </AbsoluteFill>
  );
};
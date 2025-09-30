// 玻璃态样式 - 现代磨砂玻璃效果
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




const GLASS_ACCENT = "#4F46E5"; // 靛蓝色
const GLASS_SECONDARY = "#06B6D4"; // 青蓝色

export const GlassmorphismStyle: React.FC<{
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
      {/* 背景模糊效果层 */}
      <div
        style={{
          position: "absolute",
          width: "150%",
          height: "200%",
          background: `radial-gradient(circle at 30% 20%, ${GLASS_ACCENT}40 0%, transparent 50%), radial-gradient(circle at 70% 80%, ${GLASS_SECONDARY}30 0%, transparent 50%)`,
          filter: "blur(40px)",
          opacity: 0.6,
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
          filter: "drop-shadow(0 8px 32px rgba(0, 0, 0, 0.3))",
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

            // 玻璃效果动画
            const glassShimmer = Math.sin(frame * 0.1) * 0.3 + 0.7;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? "white"
                    : hasBeenRead 
                      ? "rgba(255, 255, 255, 0.8)"
                      : "white",
                  
                  // 玻璃态背景
                  background: isCurrentlyReading 
                    ? `linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.25) 0%, 
                        rgba(255, 255, 255, 0.1) 50%, 
                        rgba(255, 255, 255, 0.05) 100%)`
                    : "transparent",
                  backdropFilter: isCurrentlyReading ? "blur(20px) saturate(180%)" : "none",
                  WebkitBackdropFilter: isCurrentlyReading ? "blur(20px) saturate(180%)" : "none",
                  borderRadius: isCurrentlyReading ? "20px" : "0px",
                  padding: isCurrentlyReading ? "8px 18px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 玻璃边框
                  border: isCurrentlyReading 
                    ? `1px solid rgba(255, 255, 255, ${0.3 * glassShimmer})` 
                    : "none",
                  
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  transition: "all 0.4s cubic-bezier(0.23, 1, 0.320, 1)",
                  
                  transform: isCurrentlyReading 
                    ? `scale(1.1) translateY(-6px)`
                    : notYetRead 
                      ? "scale(0.9) translateY(8px)"
                      : "scale(1) translateY(0px)",
                  
                  // 玻璃光泽效果
                  boxShadow: isCurrentlyReading 
                    ? `0 8px 32px rgba(79, 70, 229, 0.3), 
                       inset 0 1px 0 rgba(255, 255, 255, 0.4),
                       inset 0 -1px 0 rgba(255, 255, 255, 0.1)`
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? "0 2px 8px rgba(0, 0, 0, 0.5)"
                    : hasBeenRead
                      ? "0 1px 4px rgba(0, 0, 0, 0.3)"
                      : "0 2px 6px rgba(0, 0, 0, 0.4)",
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
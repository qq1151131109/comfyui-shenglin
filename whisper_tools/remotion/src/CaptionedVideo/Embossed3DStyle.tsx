// 立体浮凸样式 - 3D浮雕效果
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





export const Embossed3DStyle: React.FC<{
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
          color: "#E5E7EB",
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

            // 3D倾斜角度
            const tiltAngle = Math.sin(frame * 0.05) * 5;

            return (
              <span
                key={t.fromMs}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  position: "relative",
                  
                  opacity: notYetRead ? 0 : 1,
                  color: isCurrentlyReading 
                    ? "#FFFFFF"
                    : hasBeenRead 
                      ? "#D1D5DB"
                      : "#E5E7EB",
                  
                  // 立体背景
                  background: isCurrentlyReading 
                    ? `linear-gradient(145deg, #F3F4F6 0%, #E5E7EB 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "16px" : "0px",
                  padding: isCurrentlyReading ? "8px 18px" : "0px",
                  margin: isCurrentlyReading ? "0 5px" : "0",
                  
                  // 修复垂直对齐
                  top: isCurrentlyReading ? "-4px" : "0px",
                  
                  // 立体边框
                  border: isCurrentlyReading 
                    ? "2px solid #D1D5DB" 
                    : "none",
                  
                  transition: "all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
                  
                  // 3D变换
                  transform: isCurrentlyReading 
                    ? `scale(1.1) translateY(-8px) 
                       perspective(1000px) rotateX(${tiltAngle}deg) rotateY(${tiltAngle * 0.5}deg)`
                    : notYetRead 
                      ? "scale(0.9) translateY(6px)"
                      : "scale(1) translateY(0px)",
                  
                  // 多层立体阴影
                  boxShadow: isCurrentlyReading 
                    ? `
                       inset 2px 2px 5px rgba(255, 255, 255, 0.8),
                       inset -2px -2px 5px rgba(0, 0, 0, 0.1),
                       0 8px 16px rgba(0, 0, 0, 0.15),
                       0 4px 8px rgba(0, 0, 0, 0.1),
                       0 2px 4px rgba(0, 0, 0, 0.05)
                     `
                    : hasBeenRead
                      ? `
                         inset 1px 1px 3px rgba(255, 255, 255, 0.5),
                         inset -1px -1px 3px rgba(0, 0, 0, 0.1),
                         0 2px 4px rgba(0, 0, 0, 0.1)
                       `
                      : `
                         inset 1px 1px 2px rgba(255, 255, 255, 0.3),
                         inset -1px -1px 2px rgba(0, 0, 0, 0.15),
                         0 1px 2px rgba(0, 0, 0, 0.1)
                       `,
                  
                  // 立体文字阴影
                  textShadow: isCurrentlyReading 
                    ? `
                       1px 1px 0px rgba(0, 0, 0, 0.2),
                       2px 2px 0px rgba(0, 0, 0, 0.15),
                       3px 3px 0px rgba(0, 0, 0, 0.1),
                       0 0 10px rgba(255, 255, 255, 0.5)
                     `
                    : hasBeenRead
                      ? `
                         1px 1px 0px rgba(0, 0, 0, 0.15),
                         2px 2px 0px rgba(0, 0, 0, 0.1)
                       `
                      : `
                         1px 1px 0px rgba(0, 0, 0, 0.1),
                         0 0 5px rgba(255, 255, 255, 0.3)
                       `,
                  
                  // 字体权重
                  fontWeight: isCurrentlyReading ? "900" : "700",
                  
                  // 高光效果
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "10%",
                    left: "15%",
                    width: "70%",
                    height: "30%",
                    background: `linear-gradient(135deg, 
                      rgba(255, 255, 255, 0.6) 0%, 
                      rgba(255, 255, 255, 0.2) 50%, 
                      transparent 100%)`,
                    borderRadius: "8px",
                    filter: "blur(2px)",
                    pointerEvents: "none",
                  } : undefined,
                  
                  // 底部深度线
                  '::after': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    bottom: "2px",
                    left: "10%",
                    right: "10%",
                    height: `${height * 0.001}px`,
                    background: `linear-gradient(90deg, 
                      transparent, 
                      rgba(0, 0, 0, 0.2), 
                      transparent)`,
                    borderRadius: "1px",
                  } : undefined,
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
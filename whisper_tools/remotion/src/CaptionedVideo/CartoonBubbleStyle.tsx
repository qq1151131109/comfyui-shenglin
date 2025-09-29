// 卡通气泡样式 - 可爱的卡通气泡效果
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
const BUBBLE_COLORS = [
  "#FF6B9D", // 粉红
  "#4ECDC4", // 青绿
  "#45B7D1", // 天蓝
  "#FFA726", // 橙色
  "#AB47BC", // 紫色
  "#66BB6A", // 绿色
];

export const CartoonBubbleStyle: React.FC<{
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
      {/* 背景装饰气泡 */}
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            width: `${20 + i * 15}px`,
            height: `${20 + i * 15}px`,
            background: `${BUBBLE_COLORS[i % BUBBLE_COLORS.length]}40`,
            borderRadius: "50%",
            left: `${10 + i * 15}%`,
            top: `${20 + Math.sin(frame * 0.02 + i) * 20}%`,
            animation: `bubble-float-${i} ${3 + i}s ease-in-out infinite`,
            filter: "blur(2px)",
            opacity: 0.6,
          }}
        />
      ))}
      
      <div
        style={{
          fontSize,
          color: "white",
          fontFamily,
          textTransform: "uppercase",
          transform: makeTransform([
            scale(interpolate(enterProgress, [0, 1], [0.8, 1])),
            translateY(interpolate(enterProgress, [0, 1], [30, 0])),
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

            // 为每个词分配颜色
            const bubbleColor = BUBBLE_COLORS[index % BUBBLE_COLORS.length];
            
            // 弹跳动画
            const bounceOffset = isCurrentlyReading ? Math.sin(frame * 0.3) * 3 : 0;

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
                      ? "rgba(255, 255, 255, 0.9)"
                      : "white",
                  
                  // 卡通气泡背景
                  background: isCurrentlyReading 
                    ? `radial-gradient(ellipse at center, 
                        ${bubbleColor} 0%, 
                        ${bubbleColor}E6 30%, 
                        ${bubbleColor}CC 70%, 
                        ${bubbleColor}B3 100%)`
                    : "transparent",
                  borderRadius: isCurrentlyReading ? "50px" : "0px",
                  padding: isCurrentlyReading ? "10px 20px" : "0px",
                  margin: isCurrentlyReading ? "0 6px" : "0",
                  
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-5px" : "0px",
                  
                  // 卡通描边
                  border: isCurrentlyReading 
                    ? "4px solid white" 
                    : "none",
                  
                  transition: "all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55)",
                  
                  transform: isCurrentlyReading 
                    ? `scale(${1.2 + Math.sin(frame * 0.2) * 0.1}) translateY(${-8 + bounceOffset}px)`
                    : notYetRead 
                      ? "scale(0.7) translateY(15px)"
                      : "scale(1) translateY(0px)",
                  
                  // 卡通阴影
                  boxShadow: isCurrentlyReading 
                    ? `0 8px 25px ${bubbleColor}80, 
                       inset 0 4px 8px rgba(255, 255, 255, 0.3),
                       inset 0 -2px 4px rgba(0, 0, 0, 0.2),
                       0 0 0 8px ${bubbleColor}40`
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? `2px 2px 0px ${bubbleColor}, 
                       4px 4px 10px rgba(0, 0, 0, 0.5)`
                    : hasBeenRead
                      ? "1px 1px 3px rgba(0, 0, 0, 0.4)"
                      : "2px 2px 6px rgba(0, 0, 0, 0.6)",
                  
                  // 字体加粗
                  fontWeight: isCurrentlyReading ? "900" : "700",
                  
                  // 光泽效果
                  '::before': isCurrentlyReading ? {
                    content: '""',
                    position: "absolute",
                    top: "15%",
                    left: "20%",
                    width: "30%",
                    height: "30%",
                    background: "rgba(255, 255, 255, 0.6)",
                    borderRadius: "50%",
                    filter: "blur(4px)",
                  } : undefined,
                }}
              >
                {t.text}
              </span>
            );
          })}
        </span>
      </div>
      
      {/* CSS动画 */}
      <style>{`
        ${[1, 2, 3, 4, 5].map(i => `
          @keyframes bubble-float-${i} {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-${10 + i * 2}px) rotate(${i * 2}deg); }
            66% { transform: translateY(${5 + i}px) rotate(-${i}deg); }
          }
        `).join('')}
      `}</style>
    </AbsoluteFill>
  );
};
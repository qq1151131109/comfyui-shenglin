// 导入React核心库
import React from "react";
// 导入Remotion相关组件和钩子
import {
  AbsoluteFill,        // 绝对定位填充容器
  interpolate,         // 插值函数，用于动画过渡
  useCurrentFrame,     // 获取当前帧数的钩子
  useVideoConfig,      // 获取视频配置信息的钩子
} from "remotion";
// 导入自定义字体
import { TheBoldFont } from "../load-font";
// 导入文本适配工具，用于自动调整字体大小以适应容器宽度
import { fitText } from "@remotion/layout-utils";
// 导入动画工具函数
import { makeTransform, scale, translateY } from "@remotion/animation-utils";
// 导入TikTok页面类型定义
import { TikTokPage } from "@remotion/captions";

// 设置字体家族为自定义的粗体字体
const fontFamily = TheBoldFont;

// 定义字幕容器的样式
const container: React.CSSProperties = {
  justifyContent: "center",  // 水平居中对齐
  alignItems: "center",      // 垂直居中对齐
  top: undefined,            // 不设置顶部距离
  bottom: 350,               // 距离底部350像素（字幕位置）
  height: 150,               // 容器高度150像素
};

// 期望的字体大小（像素）
const DESIRED_FONT_SIZE = 120;
// 当前正在播放词汇的高亮颜色（亮绿色）
const HIGHLIGHT_COLOR = "#39E508";

// 导出字幕页面组件
export const Page: React.FC<{
  readonly enterProgress: number;  // 入场动画进度（0-1）
  readonly page: TikTokPage;      // TikTok字幕页面数据
}> = ({ enterProgress, page }) => {
  // 获取当前视频帧数
  const frame = useCurrentFrame();
  // 获取视频配置：宽度和帧率
  const { width, fps } = useVideoConfig();
  // 将当前帧数转换为毫秒时间戳
  const timeInMs = (frame / fps) * 1000;

  // 根据容器宽度自动适配文字大小
  const fittedText = fitText({
    fontFamily,                    // 使用的字体
    text: page.text,              // 要显示的文本内容
    withinWidth: width * 0.9,     // 限制宽度为屏幕宽度的90%
    textTransform: "uppercase",   // 转换为大写字母
  });

  // 计算最终字体大小：取期望大小和适配大小的较小值
  const fontSize = Math.min(DESIRED_FONT_SIZE, fittedText.fontSize);

  // 返回字幕组件的JSX结构
  return (
    <AbsoluteFill style={container}>
      <div
        style={{
          fontSize,                        // 动态计算的字体大小
          color: "white",                  // 基础文字颜色为白色
          WebkitTextStroke: "20px black",  // 黑色描边20像素，增强对比度
          paintOrder: "stroke",            // 描边在文字下方渲染
          // 入场动画变换：缩放和垂直移动
          transform: makeTransform([
            // 缩放动画：从80%放大到100%
            scale(interpolate(enterProgress, [0, 1], [0.8, 1])),
            // 垂直移动：从下方50像素移动到原位置
            translateY(interpolate(enterProgress, [0, 1], [50, 0])),
          ]),
          fontFamily,                      // 使用自定义字体
          textTransform: "uppercase",      // 强制转换为大写字母
        }}
      >
        <span
          style={{
            // 与父容器相同的变换动画，确保一致性
            transform: makeTransform([
              // 缩放动画：从80%放大到100%
              scale(interpolate(enterProgress, [0, 1], [0.8, 1])),
              // 垂直移动：从下方50像素移动到原位置  
              translateY(interpolate(enterProgress, [0, 1], [50, 0])),
            ]),
          }}
        >
          {/* 遍历当前页面的所有词汇token，实现逐词高亮效果 */}
          {page.tokens.map((t) => {
            // 计算词汇相对于当前序列开始时间的偏移
            const startRelativeToSequence = t.fromMs - page.startMs;
            // 计算词汇相对于当前序列结束时间的偏移
            const endRelativeToSequence = t.toMs - page.startMs;

            // 判断词汇的不同状态
            const isCurrentlyReading = 
              startRelativeToSequence <= timeInMs &&  // 已经开始
              endRelativeToSequence > timeInMs;       // 还未结束
            
            const hasBeenRead = endRelativeToSequence <= timeInMs;  // 已经读完
            const notYetRead = startRelativeToSequence > timeInMs;  // 还没开始读

            // 返回单个词汇的span元素
            return (
              <span
                key={t.fromMs}  // 使用开始时间作为唯一键
                style={{
                  display: "inline-block",     // 内联块显示，支持变换
                  whiteSpace: "pre",          // 保留空格和换行符
                  
                  // 根据状态设置透明度和颜色
                  opacity: notYetRead ? 0 : 1,  // 未读到的词汇完全透明
                  color: isCurrentlyReading 
                    ? HIGHLIGHT_COLOR           // 正在读：高亮绿色
                    : hasBeenRead 
                      ? "rgba(255, 255, 255, 0.8)"  // 已读完：半透明白色
                      : "white",                     // 默认：白色
                  
                  // 圆角背景框样式（只在正在读时显示）
                  backgroundColor: isCurrentlyReading ? "rgba(57, 229, 8, 0.3)" : "transparent",
                  borderRadius: isCurrentlyReading ? "12px" : "0px",
                  padding: isCurrentlyReading ? "4px 12px" : "0px",
                  margin: isCurrentlyReading ? "0 4px" : "0",
                  // 修复垂直对齐
                  position: "relative",
                  top: isCurrentlyReading ? "-2px" : "0px",
                  
                  // 动画过渡效果
                  transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
                  
                  // 动态变换效果
                  transform: isCurrentlyReading 
                    ? `scale(1.15) translateY(-3px)`     // 正在读：更大的缩放和上移
                    : notYetRead 
                      ? "scale(0.8) translateY(10px)"   // 未读：缩小并下移（准备出现）
                      : "scale(1) translateY(0px)",      // 已读：正常大小
                  
                  // 阴影效果
                  boxShadow: isCurrentlyReading 
                    ? "0 6px 16px rgba(57, 229, 8, 0.5)" 
                    : "none",
                  
                  // 文字阴影
                  textShadow: isCurrentlyReading 
                    ? "2px 2px 10px rgba(0, 0, 0, 0.9)"    // 正在读：强阴影
                    : hasBeenRead
                      ? "1px 1px 6px rgba(0, 0, 0, 0.7)"   // 已读：中等阴影
                      : "2px 2px 4px rgba(0, 0, 0, 0.6)",  // 默认阴影
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
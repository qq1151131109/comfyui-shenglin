import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AbsoluteFill,
  CalculateMetadataFunction,
  cancelRender,
  continueRender,
  delayRender,
  getStaticFiles,
  OffthreadVideo,
  Sequence,
  useVideoConfig,
  watchStaticFile,
  staticFile,
} from "remotion";
import { z } from "zod";
import SubtitlePage from "./SubtitlePage";
import { getVideoMetadata } from "@remotion/media-utils";
import { loadFont } from "../load-font";
import { NoCaptionFile } from "./NoCaptionFile";
import { Caption } from "@remotion/captions";

export type SubtitleProp = {
  startInSeconds: number;
  text: string;
};

export const captionedVideoSchema = z.object({
  src: z.string(),
  durationInSeconds: z.number().optional(), // 可选的时长参数
});

export const calculateCaptionedVideoMetadata: CalculateMetadataFunction<
  z.infer<typeof captionedVideoSchema>
> = async ({ props }) => {
  const fps = 30;
  // 如果传入了时长参数，使用它；否则使用默认值
  const durationInSeconds = props.durationInSeconds || 30;
  
  return {
    fps,
    durationInFrames: Math.floor(durationInSeconds * fps),
  };
};

const getFileExists = (file: string) => {
  const files = getStaticFiles();
  const fileExists = files.find((f) => {
    return f.src === file;
  });
  return Boolean(fileExists);
};

// How many captions should be displayed at a time?
// Try out:
// - 1500 to display a lot of words at a time
// - 200 to only display 1 word at a time
// 设置为1000ms，与我们的测试数据完美对应（每个词1000ms）
const SWITCH_CAPTIONS_EVERY_MS = 1000;

export const CaptionedVideo: React.FC<{
  src: string;
  durationInSeconds?: number;
}> = ({ src, durationInSeconds }) => {
  const [subtitles, setSubtitles] = useState<Caption[]>([]);
  const [handle] = useState(() => delayRender());
  const { fps } = useVideoConfig();

  // 确保src使用staticFile处理
  const videoSrc = src.startsWith('http') ? src : staticFile(src);
  
  const subtitlesFile = src
    .replace(/.mp4$/, ".json")
    .replace(/.mkv$/, ".json")
    .replace(/.mov$/, ".json")
    .replace(/.webm$/, ".json");
    
  // 确保字幕文件也使用staticFile处理
  const subtitlesUrl = subtitlesFile.startsWith('http') ? subtitlesFile : staticFile(subtitlesFile);

  const fetchSubtitles = useCallback(async () => {
    try {
      await loadFont();
      const res = await fetch(subtitlesUrl);
      const data = (await res.json()) as Caption[];
      setSubtitles(data);
      continueRender(handle);
    } catch (e) {
      cancelRender(e);
    }
  }, [handle, subtitlesUrl]);

  useEffect(() => {
    fetchSubtitles();

    const c = watchStaticFile(subtitlesFile, () => {
      fetchSubtitles();
    });

    return () => {
      c.cancel();
    };
  }, [fetchSubtitles, src, subtitlesFile]);

  const { pages } = useMemo(() => {
    // 如果没有字幕数据，返回空页面
    if (!subtitles || subtitles.length === 0) {
      return { pages: [] };
    }

    // 自定义分页逻辑：确保每个单词都能正确显示
    const customPages = [];
    let currentTime = 0;
    
    // 获取总时长
    const totalDuration = Math.max(...subtitles.map(s => s.endMs || 0));
    
    console.log("🔍 自定义分页调试信息:");
    console.log(`  - 字幕总数: ${subtitles.length}`);
    console.log(`  - 总时长: ${totalDuration}ms`);
    console.log(`  - SWITCH_CAPTIONS_EVERY_MS: ${SWITCH_CAPTIONS_EVERY_MS}`);
    
    // 按时间段创建页面
    while (currentTime < totalDuration) {
      const pageEndTime = currentTime + SWITCH_CAPTIONS_EVERY_MS;
      
      // 找到这个时间段内的所有词汇
      const wordsInPage = subtitles.filter(subtitle => {
        const startMs = subtitle.startMs || 0;
        const endMs = subtitle.endMs || 0;
        // 词汇与页面时间段有重叠
        return startMs < pageEndTime && endMs > currentTime;
      });
      
      if (wordsInPage.length > 0) {
        // 创建页面文本
        const pageText = wordsInPage.map(w => w.text).join(' ');
        
        // 创建tokens（兼容TikTokPage格式）
        const tokens = wordsInPage.map(word => ({
          fromMs: word.startMs || 0,
          toMs: word.endMs || 0, 
          text: word.text,
          confidence: word.confidence || 1.0
        }));
        
        const page = {
          startMs: currentTime,
          endMs: Math.min(pageEndTime, totalDuration),
          text: pageText,
          tokens: tokens
        };
        
        customPages.push(page);
        console.log(`  页面 ${customPages.length - 1}: "${pageText}" (${currentTime}-${page.endMs}ms, tokens: ${tokens.length})`);
        
        // 调试每个token
        tokens.forEach((token, tokenIndex) => {
          console.log(`    token ${tokenIndex}: "${token.text}" (${token.fromMs}-${token.toMs}ms)`);
        });
      }
      
      currentTime += SWITCH_CAPTIONS_EVERY_MS;
      
      // 防止无限循环
      if (customPages.length > 100) {
        console.log("⚠️ 达到最大页面数限制，停止分页");
        break;
      }
    }
    
    console.log(`  - 最终分页数: ${customPages.length}`);
    
    return { pages: customPages };
  }, [subtitles]);

  return (
    <AbsoluteFill style={{ backgroundColor: "white" }}>
      <AbsoluteFill>
        <OffthreadVideo
          style={{
            objectFit: "cover",
          }}
          src={videoSrc}
        />
      </AbsoluteFill>
      {pages.map((page, index) => {
        const nextPage = pages[index + 1] ?? null;
        const subtitleStartFrame = (page.startMs / 1000) * fps;
        
        // 修复：直接使用页面的endMs，而不是复杂计算
        const subtitleEndFrame = (page.endMs / 1000) * fps;
        const durationInFrames = subtitleEndFrame - subtitleStartFrame;
        
        // 调试信息：输出Sequence参数
        console.log(`🎬 Sequence ${index}:`, {
          text: page.text,
          startMs: page.startMs,
          endMs: page.endMs, 
          startFrame: subtitleStartFrame,
          endFrame: subtitleEndFrame,
          durationFrames: durationInFrames,
          nextPageStartMs: nextPage?.startMs || 'none'
        });
        
        if (durationInFrames <= 0) {
          console.log(`⚠️  跳过页面 ${index}: 持续时间 <= 0`);
          return null;
        }

        return (
          <Sequence
            key={index}
            from={subtitleStartFrame}
            durationInFrames={durationInFrames}
          >
            <SubtitlePage key={index} page={page} />;
          </Sequence>
        );
      })}
      {subtitles && subtitles.length > 0 ? null : <NoCaptionFile />}
    </AbsoluteFill>
  );
};

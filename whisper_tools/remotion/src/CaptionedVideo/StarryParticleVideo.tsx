import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { StarryParticleStyle } from "./StarryParticleStyle";

// 重新导出通用schema和metadata函数
export const starryParticleVideoSchema = genericCaptionedVideoSchema;
export const calculateStarryParticleVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const StarryParticleVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={StarryParticleStyle}
    />
  );
};
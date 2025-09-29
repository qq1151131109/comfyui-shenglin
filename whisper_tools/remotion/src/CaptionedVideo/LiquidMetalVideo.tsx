import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { LiquidMetalStyle } from "./LiquidMetalStyle";

export const liquidMetalVideoSchema = genericCaptionedVideoSchema;
export const calculateLiquidMetalVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const LiquidMetalVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={LiquidMetalStyle}
    />
  );
};
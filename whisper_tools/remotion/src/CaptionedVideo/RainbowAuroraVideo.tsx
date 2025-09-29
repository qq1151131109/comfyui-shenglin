import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { RainbowAuroraStyle } from "./RainbowAuroraStyle";

export const rainbowAuroraVideoSchema = genericCaptionedVideoSchema;
export const calculateRainbowAuroraVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const RainbowAuroraVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={RainbowAuroraStyle}
    />
  );
};
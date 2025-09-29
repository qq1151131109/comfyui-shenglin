import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { WaterRippleStyle } from "./WaterRippleStyle";

export const waterRippleVideoSchema = genericCaptionedVideoSchema;
export const calculateWaterRippleVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const WaterRippleVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={WaterRippleStyle}
    />
  );
};
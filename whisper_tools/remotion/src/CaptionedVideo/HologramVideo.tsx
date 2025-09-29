import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { HologramStyle } from "./HologramStyle";

export const hologramVideoSchema = genericCaptionedVideoSchema;
export const calculateHologramVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const HologramVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={HologramStyle}
    />
  );
};
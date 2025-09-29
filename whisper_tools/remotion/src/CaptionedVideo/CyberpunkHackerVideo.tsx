import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { CyberpunkHackerStyle } from "./CyberpunkHackerStyle";

export const cyberpunkHackerVideoSchema = genericCaptionedVideoSchema;
export const calculateCyberpunkHackerVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const CyberpunkHackerVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={CyberpunkHackerStyle}
    />
  );
};
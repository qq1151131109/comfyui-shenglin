import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { SmokeMistStyle } from "./SmokeMistStyle";

export const smokeMistVideoSchema = genericCaptionedVideoSchema;
export const calculateSmokeMistVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const SmokeMistVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={SmokeMistStyle}
    />
  );
};
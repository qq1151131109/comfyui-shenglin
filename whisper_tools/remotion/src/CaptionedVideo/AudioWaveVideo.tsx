import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { AudioWaveStyle } from "./AudioWaveStyle";

export const audioWaveVideoSchema = genericCaptionedVideoSchema;
export const calculateAudioWaveVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const AudioWaveVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={AudioWaveStyle}
    />
  );
};
import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { MagicSpellStyle } from "./MagicSpellStyle";

export const magicSpellVideoSchema = genericCaptionedVideoSchema;
export const calculateMagicSpellVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const MagicSpellVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={MagicSpellStyle}
    />
  );
};
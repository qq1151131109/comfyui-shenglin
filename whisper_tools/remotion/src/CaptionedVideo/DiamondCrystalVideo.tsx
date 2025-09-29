import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { DiamondCrystalStyle } from "./DiamondCrystalStyle";

export const diamondCrystalVideoSchema = genericCaptionedVideoSchema;
export const calculateDiamondCrystalVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const DiamondCrystalVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={DiamondCrystalStyle}
    />
  );
};
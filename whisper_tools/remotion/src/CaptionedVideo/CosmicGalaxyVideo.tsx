import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { CosmicGalaxyStyle } from "./CosmicGalaxyStyle";

export const cosmicGalaxyVideoSchema = genericCaptionedVideoSchema;
export const calculateCosmicGalaxyVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const CosmicGalaxyVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={CosmicGalaxyStyle}
    />
  );
};
import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { SpaceTimeWarpStyle } from "./SpaceTimeWarpStyle";

export const spaceTimeWarpVideoSchema = genericCaptionedVideoSchema;
export const calculateSpaceTimeWarpVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const SpaceTimeWarpVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={SpaceTimeWarpStyle}
    />
  );
};
import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { ElectricLightningStyle } from "./ElectricLightningStyle";

export const electricLightningVideoSchema = genericCaptionedVideoSchema;
export const calculateElectricLightningVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const ElectricLightningVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={ElectricLightningStyle}
    />
  );
};
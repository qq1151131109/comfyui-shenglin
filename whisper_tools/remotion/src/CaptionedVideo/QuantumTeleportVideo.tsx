import {
  GenericCaptionedVideo,
  genericCaptionedVideoSchema,
  calculateGenericCaptionedVideoMetadata,
} from "./GenericCaptionedVideo";
import { QuantumTeleportStyle } from "./QuantumTeleportStyle";

export const quantumTeleportVideoSchema = genericCaptionedVideoSchema;
export const calculateQuantumTeleportVideoMetadata = calculateGenericCaptionedVideoMetadata;

export const QuantumTeleportVideo: React.FC<{
  src: string;
}> = ({ src }) => {
  return (
    <GenericCaptionedVideo
      src={src}
      StyleComponent={QuantumTeleportStyle}
    />
  );
};
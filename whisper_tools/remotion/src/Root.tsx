import { Composition, staticFile } from "remotion";
import {
  CaptionedVideo,
  calculateCaptionedVideoMetadata,
  captionedVideoSchema,
} from "./CaptionedVideo";
import {
  NeonGlowVideo,
  calculateNeonGlowVideoMetadata,
  neonGlowVideoSchema,
} from "./CaptionedVideo/NeonGlowVideo";
import {
  RetroWaveVideo,
  calculateRetroWaveVideoMetadata,
  retroWaveVideoSchema,
} from "./CaptionedVideo/RetroWaveVideo";
import {
  GlitchVideo,
  calculateGlitchVideoMetadata,
  glitchVideoSchema,
} from "./CaptionedVideo/GlitchVideo";
import {
  MinimalVideo,
  calculateMinimalVideoMetadata,
  minimalVideoSchema,
} from "./CaptionedVideo/MinimalVideo";
// 新增样式导入
import {
  GlassmorphismVideo,
  calculateGlassmorphismVideoMetadata,
  glassmorphismVideoSchema,
} from "./CaptionedVideo/GlassmorphismVideo";
import {
  GradientRainbowVideo,
  calculateGradientRainbowVideoMetadata,
  gradientRainbowVideoSchema,
} from "./CaptionedVideo/GradientRainbowVideo";
import {
  CartoonBubbleVideo,
  calculateCartoonBubbleVideoMetadata,
  cartoonBubbleVideoSchema,
} from "./CaptionedVideo/CartoonBubbleVideo";
import {
  MetallicVideo,
  calculateMetallicVideoMetadata,
  metallicVideoSchema,
} from "./CaptionedVideo/MetallicVideo";
import {
  NeonBorderVideo,
  calculateNeonBorderVideoMetadata,
  neonBorderVideoSchema,
} from "./CaptionedVideo/NeonBorderVideo";
import {
  PixelArtVideo,
  calculatePixelArtVideoMetadata,
  pixelArtVideoSchema,
} from "./CaptionedVideo/PixelArtVideo";
import {
  ElegantShadowVideo,
  calculateElegantShadowVideoMetadata,
  elegantShadowVideoSchema,
} from "./CaptionedVideo/ElegantShadowVideo";
import {
  Embossed3DVideo,
  calculateEmbossed3DVideoMetadata,
  embossed3DVideoSchema,
} from "./CaptionedVideo/Embossed3DVideo";
import {
  FluidGradientVideo,
  calculateFluidGradientVideoMetadata,
  fluidGradientVideoSchema,
} from "./CaptionedVideo/FluidGradientVideo";
import {
  TechWireframeVideo,
  calculateTechWireframeVideoMetadata,
  techWireframeVideoSchema,
} from "./CaptionedVideo/TechWireframeVideo";
import {
  BouncyBallVideo,
  calculateBouncyBallVideoMetadata,
  bouncyBallVideoSchema,
} from "./CaptionedVideo/BouncyBallVideo";
import {
  SpinSpiralVideo,
  calculateSpinSpiralVideoMetadata,
  spinSpiralVideoSchema,
} from "./CaptionedVideo/SpinSpiralVideo";
import {
  ElasticZoomVideo,
  calculateElasticZoomVideoMetadata,
  elasticZoomVideoSchema,
} from "./CaptionedVideo/ElasticZoomVideo";
import {
  SwayBeatVideo,
  calculateSwayBeatVideoMetadata,
  swayBeatVideoSchema,
} from "./CaptionedVideo/SwayBeatVideo";
import {
  ShakeImpactVideo,
  calculateShakeImpactVideoMetadata,
  shakeImpactVideoSchema,
} from "./CaptionedVideo/ShakeImpactVideo";
import {
  FireFlameVideo,
  calculateFireFlameVideoMetadata,
  fireFlameVideoSchema,
} from "./CaptionedVideo/FireFlameVideo";
import {
  WaterRippleVideo,
  calculateWaterRippleVideoMetadata,
  waterRippleVideoSchema,
} from "./CaptionedVideo/WaterRippleVideo";
import {
  ElectricLightningVideo,
  calculateElectricLightningVideoMetadata,
  electricLightningVideoSchema,
} from "./CaptionedVideo/ElectricLightningVideo";
import {
  SmokeMistVideo,
  calculateSmokeMistVideoMetadata,
  smokeMistVideoSchema,
} from "./CaptionedVideo/SmokeMistVideo";
import {
  StarryParticleVideo,
  calculateStarryParticleVideoMetadata,
  starryParticleVideoSchema,
} from "./CaptionedVideo/StarryParticleVideo";
// 超级炫酷样式导入
import {
  QuantumTeleportVideo,
  calculateQuantumTeleportVideoMetadata,
  quantumTeleportVideoSchema,
} from "./CaptionedVideo/QuantumTeleportVideo";
import {
  HologramVideo,
  calculateHologramVideoMetadata,
  hologramVideoSchema,
} from "./CaptionedVideo/HologramVideo";
import {
  MagicSpellVideo,
  calculateMagicSpellVideoMetadata,
  magicSpellVideoSchema,
} from "./CaptionedVideo/MagicSpellVideo";
import {
  CyberpunkHackerVideo,
  calculateCyberpunkHackerVideoMetadata,
  cyberpunkHackerVideoSchema,
} from "./CaptionedVideo/CyberpunkHackerVideo";
import {
  CosmicGalaxyVideo,
  calculateCosmicGalaxyVideoMetadata,
  cosmicGalaxyVideoSchema,
} from "./CaptionedVideo/CosmicGalaxyVideo";
import {
  SpaceTimeWarpVideo,
  calculateSpaceTimeWarpVideoMetadata,
  spaceTimeWarpVideoSchema,
} from "./CaptionedVideo/SpaceTimeWarpVideo";
import {
  DiamondCrystalVideo,
  calculateDiamondCrystalVideoMetadata,
  diamondCrystalVideoSchema,
} from "./CaptionedVideo/DiamondCrystalVideo";
import {
  AudioWaveVideo,
  calculateAudioWaveVideoMetadata,
  audioWaveVideoSchema,
} from "./CaptionedVideo/AudioWaveVideo";
import {
  LiquidMetalVideo,
  calculateLiquidMetalVideoMetadata,
  liquidMetalVideoSchema,
} from "./CaptionedVideo/LiquidMetalVideo";
import {
  RainbowAuroraVideo,
  calculateRainbowAuroraVideoMetadata,
  rainbowAuroraVideoSchema,
} from "./CaptionedVideo/RainbowAuroraVideo";

// Each <Composition> is an entry in the sidebar!

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="CaptionedVideo"
        component={CaptionedVideo}
        calculateMetadata={calculateCaptionedVideoMetadata}
        schema={captionedVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="NeonGlowStyle"
        component={NeonGlowVideo}
        calculateMetadata={calculateNeonGlowVideoMetadata}
        schema={neonGlowVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="RetroWaveStyle"
        component={RetroWaveVideo}
        calculateMetadata={calculateRetroWaveVideoMetadata}
        schema={retroWaveVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="GlitchStyle"
        component={GlitchVideo}
        calculateMetadata={calculateGlitchVideoMetadata}
        schema={glitchVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="MinimalStyle"
        component={MinimalVideo}
        calculateMetadata={calculateMinimalVideoMetadata}
        schema={minimalVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="GlassmorphismStyle"
        component={GlassmorphismVideo}
        calculateMetadata={calculateGlassmorphismVideoMetadata}
        schema={glassmorphismVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="GradientRainbowStyle"
        component={GradientRainbowVideo}
        calculateMetadata={calculateGradientRainbowVideoMetadata}
        schema={gradientRainbowVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="CartoonBubbleStyle"
        component={CartoonBubbleVideo}
        calculateMetadata={calculateCartoonBubbleVideoMetadata}
        schema={cartoonBubbleVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="MetallicStyle"
        component={MetallicVideo}
        calculateMetadata={calculateMetallicVideoMetadata}
        schema={metallicVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="NeonBorderStyle"
        component={NeonBorderVideo}
        calculateMetadata={calculateNeonBorderVideoMetadata}
        schema={neonBorderVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="PixelArtStyle"
        component={PixelArtVideo}
        calculateMetadata={calculatePixelArtVideoMetadata}
        schema={pixelArtVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="ElegantShadowStyle"
        component={ElegantShadowVideo}
        calculateMetadata={calculateElegantShadowVideoMetadata}
        schema={elegantShadowVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="Embossed3DStyle"
        component={Embossed3DVideo}
        calculateMetadata={calculateEmbossed3DVideoMetadata}
        schema={embossed3DVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="FluidGradientStyle"
        component={FluidGradientVideo}
        calculateMetadata={calculateFluidGradientVideoMetadata}
        schema={fluidGradientVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="TechWireframeStyle"
        component={TechWireframeVideo}
        calculateMetadata={calculateTechWireframeVideoMetadata}
        schema={techWireframeVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="BouncyBallStyle"
        component={BouncyBallVideo}
        calculateMetadata={calculateBouncyBallVideoMetadata}
        schema={bouncyBallVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="SpinSpiralStyle"
        component={SpinSpiralVideo}
        calculateMetadata={calculateSpinSpiralVideoMetadata}
        schema={spinSpiralVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="ElasticZoomStyle"
        component={ElasticZoomVideo}
        calculateMetadata={calculateElasticZoomVideoMetadata}
        schema={elasticZoomVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="SwayBeatStyle"
        component={SwayBeatVideo}
        calculateMetadata={calculateSwayBeatVideoMetadata}
        schema={swayBeatVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="ShakeImpactStyle"
        component={ShakeImpactVideo}
        calculateMetadata={calculateShakeImpactVideoMetadata}
        schema={shakeImpactVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="FireFlameStyle"
        component={FireFlameVideo}
        calculateMetadata={calculateFireFlameVideoMetadata}
        schema={fireFlameVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="WaterRippleStyle"
        component={WaterRippleVideo}
        calculateMetadata={calculateWaterRippleVideoMetadata}
        schema={waterRippleVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="ElectricLightningStyle"
        component={ElectricLightningVideo}
        calculateMetadata={calculateElectricLightningVideoMetadata}
        schema={electricLightningVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="SmokeMistStyle"
        component={SmokeMistVideo}
        calculateMetadata={calculateSmokeMistVideoMetadata}
        schema={smokeMistVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="StarryParticleStyle"
        component={StarryParticleVideo}
        calculateMetadata={calculateStarryParticleVideoMetadata}
        schema={starryParticleVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="QuantumTeleportStyle"
        component={QuantumTeleportVideo}
        calculateMetadata={calculateQuantumTeleportVideoMetadata}
        schema={quantumTeleportVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="HologramStyle"
        component={HologramVideo}
        calculateMetadata={calculateHologramVideoMetadata}
        schema={hologramVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="MagicSpellStyle"
        component={MagicSpellVideo}
        calculateMetadata={calculateMagicSpellVideoMetadata}
        schema={magicSpellVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="CyberpunkHackerStyle"
        component={CyberpunkHackerVideo}
        calculateMetadata={calculateCyberpunkHackerVideoMetadata}
        schema={cyberpunkHackerVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="CosmicGalaxyStyle"
        component={CosmicGalaxyVideo}
        calculateMetadata={calculateCosmicGalaxyVideoMetadata}
        schema={cosmicGalaxyVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="SpaceTimeWarpStyle"
        component={SpaceTimeWarpVideo}
        calculateMetadata={calculateSpaceTimeWarpVideoMetadata}
        schema={spaceTimeWarpVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="DiamondCrystalStyle"
        component={DiamondCrystalVideo}
        calculateMetadata={calculateDiamondCrystalVideoMetadata}
        schema={diamondCrystalVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="AudioWaveStyle"
        component={AudioWaveVideo}
        calculateMetadata={calculateAudioWaveVideoMetadata}
        schema={audioWaveVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="LiquidMetalStyle"
        component={LiquidMetalVideo}
        calculateMetadata={calculateLiquidMetalVideoMetadata}
        schema={liquidMetalVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
      <Composition
        id="RainbowAuroraStyle"
        component={RainbowAuroraVideo}
        calculateMetadata={calculateRainbowAuroraVideoMetadata}
        schema={rainbowAuroraVideoSchema}
        width={1080}
        height={1920}
        fps={60}
        defaultProps={{
          src: staticFile("sample-video.mp4"),
        }}
      />
    </>
  );
};
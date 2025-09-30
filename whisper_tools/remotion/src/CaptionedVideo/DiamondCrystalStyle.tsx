import { AbsoluteFill, useCurrentFrame } from "remotion";

interface DiamondCrystalStyleProps {
  page: {
    tokens: Array<{
      text: string;
      ts: number;
      end: number;
    }>;
  };
  enterProgress: number;
}

export const DiamondCrystalStyle: React.FC<DiamondCrystalStyleProps> = ({
  page,
  enterProgress,
}) => {
  const frame = useCurrentFrame();
  
  return (
    <AbsoluteFill style={{ 
      justifyContent: "center", 
      alignItems: "center",
      padding: 20,
      background: "radial-gradient(circle, rgba(40,0,60,0.3), rgba(0,0,0,0.8) 70%)",
    }}>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: 25,
        maxWidth: "90%",
        position: 'relative',
      }}>
        {page.tokens.map((token, tokenIndex) => {
          const isCurrentlyReading = tokenIndex / page.tokens.length <= enterProgress;
          const crystalRotation = frame * 0.8 + tokenIndex * 30;
          const prismEffect = Math.sin((frame + tokenIndex * 15) * 0.05) * 10;
          
          return (
            <div
              key={tokenIndex}
              style={{
                position: 'relative',
                transform: `rotateY(${prismEffect}deg) rotateX(${prismEffect * 0.3}deg)`,
                transformStyle: 'preserve-3d',
              }}
            >
              {/* 钻石切面背景 */}
              {isCurrentlyReading && (
                <div style={{
                  position: 'absolute',
                  inset: -30,
                  background: `
                    linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%),
                    linear-gradient(-45deg, transparent 30%, rgba(255,0,128,0.1) 50%, transparent 70%),
                    linear-gradient(135deg, transparent 30%, rgba(0,255,255,0.1) 50%, transparent 70%),
                    linear-gradient(-135deg, transparent 30%, rgba(255,255,0,0.1) 50%, transparent 70%)
                  `,
                  clipPath: 'polygon(50% 0%, 0% 50%, 50% 100%, 100% 50%)',
                  pointerEvents: 'none',
                  transform: `rotate(${crystalRotation}deg)`,
                }} />
              )}
              
              <span style={{
                fontFamily: "Raleway, sans-serif",
                fontSize: 46,
                fontWeight: 800,
                background: isCurrentlyReading
                  ? `linear-gradient(45deg, 
                      #ffffff 0%, 
                      #ff69b4 20%, 
                      #00ffff 40%, 
                      #ffff00 60%, 
                      #ff1493 80%, 
                      #ffffff 100%)`
                  : "#ffffff",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: isCurrentlyReading ? "transparent" : "#ffffff",
                backgroundClip: "text",
                textShadow: isCurrentlyReading
                  ? `0 0 30px rgba(255,255,255,0.8), 0 0 60px rgba(255,105,180,0.6),
                     -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000`
                  : "-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000",
                padding: isCurrentlyReading ? "15px 20px" : "8px 12px",
                borderRadius: "0px",
                clipPath: isCurrentlyReading 
                  ? 'polygon(10% 0%, 90% 0%, 100% 25%, 100% 75%, 90% 100%, 10% 100%, 0% 75%, 0% 25%)' 
                  : 'none',
                position: 'relative',
                top: isCurrentlyReading ? "-2px" : "0px",
                display: 'inline-block',
                background: isCurrentlyReading
                  ? `linear-gradient(135deg, 
                      rgba(255,255,255,0.15), 
                      rgba(255,105,180,0.1), 
                      rgba(0,255,255,0.1), 
                      rgba(255,255,255,0.15))`
                  : 'transparent',
                border: isCurrentlyReading ? '2px solid rgba(255,255,255,0.3)' : 'none',
                backdropFilter: isCurrentlyReading ? 'blur(2px)' : 'none',
              }}>
                {token.text}
              </span>

              {/* 水晶内部反射 */}
              {isCurrentlyReading && (
                <>
                  {Array.from({length: 8}).map((_, i) => (
                    <div
                      key={i}
                      style={{
                        position: 'absolute',
                        left: '50%',
                        top: '50%',
                        width: '2px',
                        height: `${height * (0.016 + i * 0.005)}px`,
                        background: `linear-gradient(
                          to bottom,
                          transparent,
                          hsl(${i * 45 + frame * 2}, 80%, 70%) 50%,
                          transparent
                        )`,
                        transform: `
                          translate(-50%, -50%) 
                          rotate(${i * 45 + crystalRotation}deg)
                        `,
                        opacity: 0.6,
                        filter: 'blur(0.5px)',
                        pointerEvents: 'none',
                      }}
                    />
                  ))}
                  
                  {/* 光谱分解效果 */}
                  <div style={{
                    position: 'absolute',
                    inset: -5,
                    background: `conic-gradient(
                      from ${frame * 2}deg,
                      transparent 0deg,
                      rgba(255,0,0,0.3) 60deg,
                      rgba(255,165,0,0.3) 120deg,
                      rgba(255,255,0,0.3) 180deg,
                      rgba(0,255,0,0.3) 240deg,
                      rgba(0,0,255,0.3) 300deg,
                      rgba(128,0,128,0.3) 360deg
                    )`,
                    clipPath: 'polygon(10% 0%, 90% 0%, 100% 25%, 100% 75%, 90% 100%, 10% 100%, 0% 75%, 0% 25%)',
                    filter: 'blur(8px)',
                    opacity: 0.4,
                    pointerEvents: 'none',
                  }} />
                </>
              )}

              {/* 钻石闪烁点 */}
              {isCurrentlyReading && Array.from({length: 12}).map((_, i) => {
                const sparkleSize = 2 + Math.sin((frame + i * 10) * 0.1) * 2;
                const sparkleOpacity = Math.abs(Math.sin((frame + i * 8) * 0.08));
                
                return (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      left: `${20 + (i * 7) % 60}%`,
                      top: `${15 + (i * 13) % 70}%`,
                      width: `${sparkleSize}px`,
                      height: `${sparkleSize}px`,
                      background: '#ffffff',
                      borderRadius: '50%',
                      opacity: sparkleOpacity,
                      boxShadow: `0 0 ${sparkleSize * 3}px #ffffff`,
                      pointerEvents: 'none',
                      transform: `scale(${sparkleOpacity})`,
                    }}
                  />
                );
              })}

              {/* 折射光束 */}
              {isCurrentlyReading && Math.sin(frame * 0.1 + tokenIndex) > 0.7 && (
                <>
                  {Array.from({length: 3}).map((_, i) => (
                    <div
                      key={i}
                      style={{
                        position: 'absolute',
                        left: '50%',
                        top: '50%',
                        width: '2px',
                        height: '150px',
                        background: `linear-gradient(
                          to bottom,
                          transparent,
                          hsl(${i * 120 + frame}, 90%, 70%) 30%,
                          hsl(${i * 120 + frame + 60}, 90%, 70%) 70%,
                          transparent
                        )`,
                        transform: `
                          translate(-50%, -50%) 
                          rotate(${i * 120 + frame * 3}deg)
                          translateY(-75px)
                        `,
                        opacity: 0.8,
                        filter: 'blur(1px)',
                        pointerEvents: 'none',
                      }}
                    />
                  ))}
                </>
              )}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
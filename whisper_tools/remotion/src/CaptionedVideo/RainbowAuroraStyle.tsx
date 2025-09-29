import { AbsoluteFill, useCurrentFrame } from "remotion";

interface RainbowAuroraStyleProps {
  page: {
    tokens: Array<{
      text: string;
      ts: number;
      end: number;
    }>;
  };
  enterProgress: number;
}

export const RainbowAuroraStyle: React.FC<RainbowAuroraStyleProps> = ({
  page,
  enterProgress,
}) => {
  const frame = useCurrentFrame();
  
  return (
    <AbsoluteFill style={{ 
      justifyContent: "center", 
      alignItems: "center",
      padding: 20,
      background: "radial-gradient(ellipse, rgba(0,20,40,0.3), rgba(0,5,15,0.9) 70%)",
    }}>
      {/* 极光背景 */}
      <div style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
      }}>
        {Array.from({length: 6}).map((_, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: `${i * 15 - 10}%`,
              top: '10%',
              width: '25%',
              height: '80%',
              background: `linear-gradient(
                180deg,
                transparent 0%,
                hsla(${120 + i * 60 + frame * 0.5}, 70%, 60%, 0.3) 20%,
                hsla(${180 + i * 40 + frame * 0.3}, 80%, 70%, 0.4) 50%,
                hsla(${240 + i * 30 + frame * 0.7}, 90%, 80%, 0.2) 80%,
                transparent 100%
              )`,
              transform: `
                skewX(${Math.sin((frame + i * 30) * 0.02) * 15}deg)
                translateX(${Math.sin((frame + i * 20) * 0.01) * 30}px)
              `,
              filter: 'blur(15px)',
              opacity: 0.6 + Math.sin((frame + i * 10) * 0.03) * 0.2,
            }}
          />
        ))}
      </div>
      
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: 26,
        maxWidth: "90%",
        position: 'relative',
        zIndex: 10,
      }}>
        {page.tokens.map((token, tokenIndex) => {
          const isCurrentlyReading = tokenIndex / page.tokens.length <= enterProgress;
          const auroraDance = Math.sin((frame + tokenIndex * 25) * 0.06) * 3;
          const colorShift = (frame + tokenIndex * 40) * 2;
          
          return (
            <div
              key={tokenIndex}
              style={{
                position: 'relative',
                transform: `translateY(${auroraDance}px) rotate(${auroraDance * 0.2}deg)`,
              }}
            >
              {/* 极光光带背景 */}
              {isCurrentlyReading && (
                <div style={{
                  position: 'absolute',
                  inset: -20,
                  background: `linear-gradient(
                    45deg,
                    hsla(${colorShift}, 80%, 60%, 0.2) 0%,
                    hsla(${colorShift + 60}, 70%, 70%, 0.3) 25%,
                    hsla(${colorShift + 120}, 85%, 65%, 0.2) 50%,
                    hsla(${colorShift + 180}, 75%, 75%, 0.3) 75%,
                    hsla(${colorShift + 240}, 90%, 55%, 0.2) 100%
                  )`,
                  borderRadius: '30px',
                  filter: 'blur(20px)',
                  pointerEvents: 'none',
                  animation: 'aurora-flow 4s ease-in-out infinite',
                }} />
              )}
              
              <span style={{
                fontFamily: "Comfortaa, cursive",
                fontSize: 48,
                fontWeight: 700,
                background: isCurrentlyReading
                  ? `linear-gradient(45deg, 
                      hsl(${colorShift}, 80%, 80%) 0%,
                      hsl(${colorShift + 72}, 90%, 85%) 14%,
                      hsl(${colorShift + 144}, 85%, 75%) 28%,
                      hsl(${colorShift + 216}, 95%, 90%) 42%,
                      hsl(${colorShift + 288}, 75%, 80%) 57%,
                      hsl(${colorShift}, 80%, 85%) 71%,
                      hsl(${colorShift + 72}, 90%, 75%) 85%,
                      hsl(${colorShift + 144}, 85%, 90%) 100%)`
                  : "#ffffff",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: isCurrentlyReading ? "transparent" : "#ffffff",
                backgroundClip: "text",
                backgroundSize: isCurrentlyReading ? "400% 400%" : "100% 100%",
                backgroundPosition: isCurrentlyReading 
                  ? `${50 + Math.sin(frame * 0.02) * 50}% ${50 + Math.cos(frame * 0.03) * 50}%`
                  : "center",
                textShadow: isCurrentlyReading
                  ? `0 0 30px hsla(${colorShift}, 80%, 70%, 0.8), 
                     0 0 60px hsla(${colorShift + 120}, 70%, 80%, 0.6),
                     -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000`
                  : "-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000",
                padding: isCurrentlyReading ? "15px 22px" : "8px 12px",
                borderRadius: "25px",
                border: isCurrentlyReading 
                  ? `3px solid hsla(${colorShift}, 70%, 70%, 0.5)` 
                  : "none",
                position: 'relative',
                top: isCurrentlyReading ? "-2px" : "0px",
                display: 'inline-block',
                background: isCurrentlyReading
                  ? `linear-gradient(135deg, 
                      hsla(${colorShift}, 50%, 50%, 0.15), 
                      hsla(${colorShift + 120}, 60%, 60%, 0.1), 
                      hsla(${colorShift + 240}, 70%, 70%, 0.15))`
                  : 'transparent',
                backdropFilter: isCurrentlyReading ? "blur(2px)" : "none",
                letterSpacing: "1px",
                animation: isCurrentlyReading ? 'rainbow-shimmer 3s ease-in-out infinite' : 'none',
              }}>
                {token.text}
              </span>

              {/* 极光粒子 */}
              {isCurrentlyReading && Array.from({length: 15}).map((_, i) => {
                const particleHue = (colorShift + i * 24) % 360;
                const particleSize = 2 + Math.sin((frame + i * 12) * 0.08) * 2;
                const orbitRadius = 40 + i * 3;
                const orbitSpeed = 0.04 + i * 0.002;
                
                return (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      left: '50%',
                      top: '50%',
                      width: `${particleSize}px`,
                      height: `${particleSize}px`,
                      background: `hsl(${particleHue}, 85%, 75%)`,
                      borderRadius: '50%',
                      transform: `
                        translate(-50%, -50%) 
                        rotate(${(frame + i * 24) * orbitSpeed}deg) 
                        translateY(-${orbitRadius}px)
                      `,
                      opacity: 0.6 + Math.sin((frame + i * 8) * 0.06) * 0.4,
                      boxShadow: `0 0 ${particleSize * 4}px hsl(${particleHue}, 85%, 75%)`,
                      pointerEvents: 'none',
                    }}
                  />
                );
              })}

              {/* 极光波纹 */}
              {isCurrentlyReading && (
                <>
                  <div style={{
                    position: 'absolute',
                    inset: -15,
                    border: `2px solid hsla(${colorShift}, 70%, 70%, 0.4)`,
                    borderRadius: '30px',
                    pointerEvents: 'none',
                    animation: 'aurora-pulse 2s ease-in-out infinite',
                  }} />
                  <div style={{
                    position: 'absolute',
                    inset: -25,
                    border: `1px solid hsla(${colorShift + 120}, 60%, 80%, 0.3)`,
                    borderRadius: '35px',
                    pointerEvents: 'none',
                    animation: 'aurora-pulse 3s ease-in-out infinite reverse',
                  }} />
                </>
              )}

              {/* 彩虹光束 */}
              {isCurrentlyReading && Math.sin(frame * 0.05 + tokenIndex) > 0.8 && (
                Array.from({length: 7}).map((_, i) => (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      left: '50%',
                      top: '50%',
                      width: '3px',
                      height: '100px',
                      background: `linear-gradient(
                        to bottom,
                        transparent,
                        hsl(${i * 51.4}, 90%, 70%) 30%,
                        hsl(${i * 51.4}, 90%, 70%) 70%,
                        transparent
                      )`,
                      transform: `
                        translate(-50%, -50%) 
                        rotate(${i * 51.4 + frame}deg)
                        translateY(-50px)
                      `,
                      opacity: 0.7,
                      filter: 'blur(1px)',
                      pointerEvents: 'none',
                    }}
                  />
                ))
              )}
            </div>
          );
        })}
      </div>
      
      <style jsx>{`
        @keyframes aurora-flow {
          0%, 100% { transform: translateX(0) skewX(0deg); }
          25% { transform: translateX(10px) skewX(5deg); }
          75% { transform: translateX(-10px) skewX(-5deg); }
        }
        
        @keyframes rainbow-shimmer {
          0%, 100% { filter: hue-rotate(0deg) brightness(1); }
          50% { filter: hue-rotate(30deg) brightness(1.2); }
        }
        
        @keyframes aurora-pulse {
          0%, 100% { transform: scale(1); opacity: 0.4; }
          50% { transform: scale(1.1); opacity: 0.6; }
        }
      `}</style>
    </AbsoluteFill>
  );
};
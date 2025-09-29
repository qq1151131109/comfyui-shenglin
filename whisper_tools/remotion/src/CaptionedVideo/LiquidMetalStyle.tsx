import { AbsoluteFill, useCurrentFrame } from "remotion";

interface LiquidMetalStyleProps {
  page: {
    tokens: Array<{
      text: string;
      ts: number;
      end: number;
    }>;
  };
  enterProgress: number;
}

export const LiquidMetalStyle: React.FC<LiquidMetalStyleProps> = ({
  page,
  enterProgress,
}) => {
  const frame = useCurrentFrame();
  
  return (
    <AbsoluteFill style={{ 
      justifyContent: "center", 
      alignItems: "center",
      padding: 20,
      background: "radial-gradient(circle, rgba(40,40,60,0.5), rgba(10,10,20,0.9) 70%)",
    }}>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: 24,
        maxWidth: "90%",
        position: 'relative',
      }}>
        {page.tokens.map((token, tokenIndex) => {
          const isCurrentlyReading = tokenIndex / page.tokens.length <= enterProgress;
          const liquidFlow = Math.sin((frame + tokenIndex * 20) * 0.04) * 5;
          const mercuryShimmer = Math.sin((frame + tokenIndex * 15) * 0.08) * 0.3 + 0.7;
          
          return (
            <div
              key={tokenIndex}
              style={{
                position: 'relative',
                filter: isCurrentlyReading 
                  ? `hue-rotate(${frame * 0.5}deg) contrast(1.2)` 
                  : 'none',
              }}
            >
              <span style={{
                fontFamily: "Orbitron, sans-serif",
                fontSize: 47,
                fontWeight: 800,
                background: isCurrentlyReading
                  ? `linear-gradient(45deg, 
                      #c0c0c0 0%, 
                      #ffffff 25%, 
                      #e6e6fa 50%, 
                      #d3d3d3 75%, 
                      #a0a0a0 100%)`
                  : "#ffffff",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: isCurrentlyReading ? "transparent" : "#ffffff",
                backgroundClip: "text",
                backgroundSize: isCurrentlyReading ? "200% 200%" : "100% 100%",
                backgroundPosition: isCurrentlyReading 
                  ? `${50 + Math.sin(frame * 0.05) * 50}% ${50 + Math.cos(frame * 0.03) * 50}%`
                  : "center",
                textShadow: isCurrentlyReading
                  ? `0 0 20px rgba(192,192,192,0.8), 0 0 40px rgba(230,230,250,0.6),
                     -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000`
                  : "-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000",
                padding: isCurrentlyReading ? "14px 20px" : "8px 12px",
                borderRadius: isCurrentlyReading ? `${20 + liquidFlow}px` : "8px",
                border: isCurrentlyReading ? "3px solid rgba(192,192,192,0.6)" : "none",
                position: 'relative',
                top: isCurrentlyReading ? "-2px" : "0px",
                display: 'inline-block',
                background: isCurrentlyReading
                  ? `linear-gradient(135deg, 
                      rgba(192,192,192,0.2), 
                      rgba(230,230,250,0.15), 
                      rgba(211,211,211,0.2))`
                  : 'transparent',
                backdropFilter: isCurrentlyReading ? "blur(1px)" : "none",
                transform: `skewX(${liquidFlow * 0.3}deg)`,
                transition: "all 0.1s ease-out",
                boxShadow: isCurrentlyReading
                  ? `0 0 30px rgba(192,192,192,0.4), 
                     inset 0 0 20px rgba(230,230,250,0.2)`
                  : "none",
              }}>
                {token.text}
                
                {/* 流体反射条纹 */}
                {isCurrentlyReading && (
                  <>
                    <div style={{
                      position: 'absolute',
                      top: '20%',
                      left: `${10 + Math.sin(frame * 0.06) * 30}%`,
                      width: '60%',
                      height: '3px',
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent)',
                      borderRadius: '2px',
                      opacity: mercuryShimmer,
                      pointerEvents: 'none',
                    }} />
                    <div style={{
                      position: 'absolute',
                      top: '60%',
                      left: `${20 + Math.sin(frame * 0.04 + 1) * 25}%`,
                      width: '40%',
                      height: '2px',
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent)',
                      borderRadius: '1px',
                      opacity: mercuryShimmer * 0.7,
                      pointerEvents: 'none',
                    }} />
                  </>
                )}
              </span>

              {/* 液态水银滴 */}
              {isCurrentlyReading && Array.from({length: 8}).map((_, i) => {
                const dropSize = 3 + Math.sin((frame + i * 20) * 0.07) * 2;
                const dropY = Math.sin((frame + i * 15) * 0.05) * 50;
                
                return (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      width: `${dropSize}px`,
                      height: `${dropSize}px`,
                      background: `radial-gradient(circle, #e6e6fa, #c0c0c0)`,
                      borderRadius: '50%',
                      left: `${15 + i * 10}%`,
                      top: `${30 + dropY}%`,
                      opacity: 0.8,
                      boxShadow: `
                        0 0 ${dropSize * 2}px rgba(192,192,192,0.6),
                        inset 0 0 ${dropSize}px rgba(255,255,255,0.3)
                      `,
                      pointerEvents: 'none',
                      filter: 'blur(0.5px)',
                      transform: `scale(${0.8 + Math.sin(frame * 0.08 + i) * 0.4})`,
                    }}
                  />
                );
              })}

              {/* 金属质感边缘 */}
              {isCurrentlyReading && (
                <>
                  <div style={{
                    position: 'absolute',
                    inset: -3,
                    background: `linear-gradient(45deg, 
                      rgba(255,255,255,0.3), 
                      transparent 30%, 
                      transparent 70%, 
                      rgba(192,192,192,0.3))`,
                    borderRadius: `${23 + liquidFlow}px`,
                    pointerEvents: 'none',
                  }} />
                  
                  {/* 表面张力效果 */}
                  <div style={{
                    position: 'absolute',
                    inset: -8,
                    background: `radial-gradient(
                      ellipse at ${50 + Math.sin(frame * 0.03) * 20}% ${50 + Math.cos(frame * 0.05) * 20}%,
                      rgba(230,230,250,0.2),
                      transparent 50%
                    )`,
                    borderRadius: '50%',
                    filter: 'blur(10px)',
                    opacity: mercuryShimmer,
                    pointerEvents: 'none',
                  }} />
                </>
              )}

              {/* 流体尾迹 */}
              {isCurrentlyReading && Math.sin(frame * 0.08 + tokenIndex) > 0.6 && (
                <div style={{
                  position: 'absolute',
                  left: '100%',
                  top: '50%',
                  width: '30px',
                  height: '6px',
                  background: 'linear-gradient(90deg, rgba(192,192,192,0.6), transparent)',
                  borderRadius: '3px',
                  transform: 'translateY(-50%)',
                  opacity: 0.7,
                  pointerEvents: 'none',
                  filter: 'blur(1px)',
                }} />
              )}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
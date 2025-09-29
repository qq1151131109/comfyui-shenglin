import { AbsoluteFill, useCurrentFrame } from "remotion";

interface QuantumTeleportStyleProps {
  page: {
    tokens: Array<{
      text: string;
      ts: number;
      end: number;
    }>;
  };
  enterProgress: number;
}

export const QuantumTeleportStyle: React.FC<QuantumTeleportStyleProps> = ({
  page,
  enterProgress,
}) => {
  const frame = useCurrentFrame();
  
  return (
    <AbsoluteFill style={{ 
      justifyContent: "center", 
      alignItems: "center",
      padding: 20,
    }}>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: 12,
        maxWidth: "90%",
      }}>
        {page.tokens.map((token, tokenIndex) => {
          const isCurrentlyReading = tokenIndex / page.tokens.length <= enterProgress;
          const quantumPhase = (frame + tokenIndex * 5) * 0.1;
          
          // 量子效果：每个字符独立粒子化
          const chars = token.text.split('');
          
          return (
            <div
              key={tokenIndex}
              style={{
                display: 'inline-flex',
                position: 'relative',
              }}
            >
              {chars.map((char, charIndex) => {
                const particleOffset = Math.sin(quantumPhase + charIndex * 0.5) * (isCurrentlyReading ? 0 : 5);
                const particleOpacity = isCurrentlyReading 
                  ? 1 
                  : 0.3 + Math.abs(Math.sin(quantumPhase + charIndex * 0.3)) * 0.7;
                
                // 创建量子粒子效果
                const particles = isCurrentlyReading ? [] : Array.from({length: 3}, (_, i) => (
                  <span
                    key={`particle-${i}`}
                    style={{
                      position: 'absolute',
                      left: Math.sin(quantumPhase + i * 2) * 15,
                      top: Math.cos(quantumPhase + i * 1.5) * 10,
                      fontSize: '8px',
                      color: `hsl(${180 + i * 60}, 80%, 60%)`,
                      opacity: 0.6,
                      pointerEvents: 'none',
                    }}
                  >
                    ●
                  </span>
                ));
                
                return (
                  <span
                    key={charIndex}
                    style={{
                      position: 'relative',
                      fontFamily: "Arial Black, sans-serif",
                      fontSize: 44,
                      fontWeight: 900,
                      color: "white",
                      textShadow: isCurrentlyReading
                        ? "0 0 20px #00ffff, 0 0 40px #00ffff, -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000"
                        : "-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000",
                      background: isCurrentlyReading
                        ? `linear-gradient(45deg, 
                            rgba(0,255,255,0.3) 0%, 
                            rgba(255,0,255,0.3) 50%, 
                            rgba(0,255,255,0.3) 100%)`
                        : 'transparent',
                      padding: isCurrentlyReading ? "8px 6px" : "4px 2px",
                      borderRadius: "8px",
                      transform: `
                        translate(${particleOffset}px, ${particleOffset * 0.5}px) 
                        scale(${isCurrentlyReading ? 1.1 : 1})
                        rotateX(${isCurrentlyReading ? 0 : Math.sin(quantumPhase) * 10}deg)
                      `,
                      opacity: particleOpacity,
                      transition: isCurrentlyReading 
                        ? "all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55)"
                        : "none",
                      display: 'inline-block',
                      filter: isCurrentlyReading 
                        ? "drop-shadow(0 0 10px rgba(0,255,255,0.8))"
                        : "none",
                    }}
                  >
                    {char}
                    {particles}
                  </span>
                );
              })}
              
              {/* 量子传送门效果 */}
              {isCurrentlyReading && (
                <>
                  <div style={{
                    position: 'absolute',
                    inset: -10,
                    borderRadius: '50%',
                    background: `conic-gradient(from ${frame * 2}deg, 
                      transparent, rgba(0,255,255,0.4), transparent, 
                      rgba(255,0,255,0.4), transparent)`,
                    animation: `spin 2s linear infinite`,
                    pointerEvents: 'none',
                  }} />
                  <div style={{
                    position: 'absolute',
                    inset: -15,
                    borderRadius: '50%',
                    border: '2px solid rgba(0,255,255,0.6)',
                    animation: `pulse 1s ease-in-out infinite alternate`,
                    pointerEvents: 'none',
                  }} />
                </>
              )}
            </div>
          );
        })}
      </div>
      
      <style jsx>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          to { transform: scale(1.2); }
        }
      `}</style>
    </AbsoluteFill>
  );
};
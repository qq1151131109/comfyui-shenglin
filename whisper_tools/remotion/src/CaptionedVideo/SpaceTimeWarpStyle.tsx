import { AbsoluteFill, useCurrentFrame } from "remotion";

interface SpaceTimeWarpStyleProps {
  page: {
    tokens: Array<{
      text: string;
      ts: number;
      end: number;
    }>;
  };
  enterProgress: number;
}

export const SpaceTimeWarpStyle: React.FC<SpaceTimeWarpStyleProps> = ({
  page,
  enterProgress,
}) => {
  const frame = useCurrentFrame();
  
  return (
    <AbsoluteFill style={{ 
      justifyContent: "center", 
      alignItems: "center",
      padding: 20,
      background: "radial-gradient(circle, rgba(0,0,20,0.8), rgba(20,0,40,0.9) 70%)",
    }}>
      {/* 虫洞隧道效果 */}
      <div style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
      }}>
        {Array.from({length: 12}).map((_, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              width: `${50 + i * 40}px`,
              height: `${50 + i * 40}px`,
              border: `2px solid rgba(138, 43, 226, ${0.6 - i * 0.05})`,
              borderRadius: '50%',
              transform: `
                translate(-50%, -50%) 
                rotate(${frame * 0.5 + i * 30}deg)
                scale(${1 + Math.sin((frame + i * 20) * 0.03) * 0.1})
              `,
              filter: 'blur(1px)',
            }}
          />
        ))}
      </div>
      
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: 20,
        maxWidth: "90%",
        position: 'relative',
        zIndex: 10,
      }}>
        {page.tokens.map((token, tokenIndex) => {
          const isCurrentlyReading = tokenIndex / page.tokens.length <= enterProgress;
          const warpIntensity = isCurrentlyReading ? Math.sin(frame * 0.1 + tokenIndex) * 5 : 0;
          
          return (
            <div
              key={tokenIndex}
              style={{
                position: 'relative',
                filter: isCurrentlyReading 
                  ? `blur(${Math.abs(warpIntensity) * 0.3}px) hue-rotate(${frame * 2}deg)` 
                  : 'none',
                transform: `
                  perspective(500px) 
                  rotateY(${warpIntensity}deg) 
                  rotateX(${warpIntensity * 0.5}deg)
                  translateZ(${isCurrentlyReading ? Math.sin(frame * 0.05) * 20 : 0}px)
                `,
              }}
            >
              <span style={{
                fontFamily: "Orbitron, sans-serif",
                fontSize: 48,
                fontWeight: 800,
                background: isCurrentlyReading
                  ? `linear-gradient(45deg, #8a2be2, #ff1493, #00bfff, #8a2be2)`
                  : "#ffffff",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: isCurrentlyReading ? "transparent" : "#ffffff",
                backgroundClip: "text",
                textShadow: isCurrentlyReading
                  ? `0 0 30px rgba(138, 43, 226, 0.8), 0 0 60px rgba(255, 20, 147, 0.6),
                     -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000`
                  : "-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000",
                padding: isCurrentlyReading ? "12px 18px" : "6px 10px",
                borderRadius: "15px",
                border: isCurrentlyReading ? "2px solid rgba(138, 43, 226, 0.5)" : "none",
                position: 'relative',
                top: isCurrentlyReading ? "-2px" : "0px",
                display: 'inline-block',
                animation: isCurrentlyReading ? 'time-distort 0.5s ease-in-out infinite alternate' : 'none',
              }}>
                {token.text}
              </span>

              {isCurrentlyReading && Array.from({length: 15}).map((_, i) => (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    width: '3px',
                    height: '3px',
                    background: `hsl(${270 + i * 15}, 80%, 60%)`,
                    borderRadius: '50%',
                    left: Math.sin((frame + i * 24) * 0.08) * 100 + 50 + '%',
                    top: Math.cos((frame + i * 24) * 0.08) * 80 + 50 + '%',
                    opacity: 0.7,
                    boxShadow: `0 0 10px hsl(${270 + i * 15}, 80%, 60%)`,
                    pointerEvents: 'none',
                    transform: `scale(${0.5 + Math.sin((frame + i) * 0.1) * 0.8})`,
                  }}
                />
              ))}
            </div>
          );
        })}
      </div>
      
      <style jsx>{`
        @keyframes time-distort {
          0% { letter-spacing: 1px; }
          100% { letter-spacing: 3px; }
        }
      `}</style>
    </AbsoluteFill>
  );
};
import { AbsoluteFill, useCurrentFrame } from "remotion";

interface MagicSpellStyleProps {
  page: {
    tokens: Array<{
      text: string;
      ts: number;
      end: number;
    }>;
  };
  enterProgress: number;
}

export const MagicSpellStyle: React.FC<MagicSpellStyleProps> = ({
  page,
  enterProgress,
}) => {
  const frame = useCurrentFrame();
  
  // 神秘符文符号
  const runes = ['ᚠ', 'ᚢ', 'ᚦ', 'ᚨ', 'ᚱ', 'ᚲ', 'ᚷ', 'ᚹ', 'ᚺ', 'ᚾ', 'ᛁ', 'ᛃ', 'ᛇ', 'ᛈ', 'ᛉ', 'ᛋ'];
  const magicSymbols = ['✦', '✧', '✩', '✪', '✫', '✬', '✭', '✮', '✯', '✰', '◆', '◇', '◈', '◉', '◊'];
  
  return (
    <AbsoluteFill style={{ 
      justifyContent: "center", 
      alignItems: "center",
      padding: 20,
      background: "radial-gradient(circle, rgba(25,0,51,0.3), transparent 70%)",
    }}>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: 20,
        maxWidth: "90%",
        position: 'relative',
      }}>
        {page.tokens.map((token, tokenIndex) => {
          const isCurrentlyReading = tokenIndex / page.tokens.length <= enterProgress;
          const magicPulse = Math.sin((frame + tokenIndex * 20) * 0.08) * 0.3 + 1;
          const spellRotation = (frame + tokenIndex * 30) * 0.5;
          
          return (
            <div
              key={tokenIndex}
              style={{
                position: 'relative',
                transform: `scale(${isCurrentlyReading ? magicPulse : 1})`,
                transition: isCurrentlyReading ? "none" : "all 0.3s ease",
              }}
            >
              {/* 魔法圆环 */}
              {isCurrentlyReading && (
                <>
                  {/* 外圈符文 */}
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: '140px',
                    height: '140px',
                    transform: `translate(-50%, -50%) rotate(${spellRotation}deg)`,
                    pointerEvents: 'none',
                  }}>
                    {runes.slice(0, 8).map((rune, i) => (
                      <span
                        key={i}
                        style={{
                          position: 'absolute',
                          left: '50%',
                          top: '50%',
                          transform: `
                            rotate(${i * 45}deg) 
                            translateY(-60px) 
                            rotate(${-spellRotation}deg)
                          `,
                          transformOrigin: '0 60px',
                          fontSize: '20px',
                          color: `hsl(${280 + i * 10}, 80%, 70%)`,
                          textShadow: `0 0 10px hsl(${280 + i * 10}, 80%, 70%)`,
                          opacity: 0.8,
                        }}
                      >
                        {rune}
                      </span>
                    ))}
                  </div>
                  
                  {/* 内圈魔法符号 */}
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: '100px',
                    height: '100px',
                    transform: `translate(-50%, -50%) rotate(${-spellRotation * 0.7}deg)`,
                    pointerEvents: 'none',
                  }}>
                    {magicSymbols.slice(0, 6).map((symbol, i) => (
                      <span
                        key={i}
                        style={{
                          position: 'absolute',
                          left: '50%',
                          top: '50%',
                          transform: `
                            rotate(${i * 60}deg) 
                            translateY(-40px) 
                            rotate(${spellRotation * 0.7}deg)
                          `,
                          transformOrigin: '0 40px',
                          fontSize: '16px',
                          color: `hsl(${320 + i * 15}, 90%, 80%)`,
                          textShadow: `0 0 8px hsl(${320 + i * 15}, 90%, 80%)`,
                          opacity: 0.9,
                        }}
                      >
                        {symbol}
                      </span>
                    ))}
                  </div>
                  
                  {/* 魔法光环 */}
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: '120px',
                    height: '120px',
                    border: `2px solid rgba(147, 51, 234, ${0.3 + Math.sin(frame * 0.1) * 0.2})`,
                    borderRadius: '50%',
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'none',
                    boxShadow: `
                      0 0 20px rgba(147, 51, 234, 0.5),
                      inset 0 0 20px rgba(147, 51, 234, 0.2)
                    `,
                  }} />
                  
                  {/* 内光环 */}
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    width: '80px',
                    height: '80px',
                    border: `1px solid rgba(216, 180, 254, ${0.5 + Math.sin(frame * 0.15) * 0.3})`,
                    borderRadius: '50%',
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'none',
                    boxShadow: `
                      0 0 15px rgba(216, 180, 254, 0.6),
                      inset 0 0 15px rgba(216, 180, 254, 0.3)
                    `,
                  }} />
                </>
              )}
              
              {/* 主文字 */}
              <span style={{
                fontFamily: "Cinzel, Georgia, serif",
                fontSize: 48,
                fontWeight: 700,
                color: isCurrentlyReading ? "#e879f9" : "#ffffff",
                textShadow: isCurrentlyReading
                  ? `0 0 20px #e879f9, 0 0 40px #c084fc, 0 0 60px #9333ea,
                     -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000`
                  : "-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000",
                background: isCurrentlyReading
                  ? `linear-gradient(45deg, 
                      rgba(147, 51, 234, 0.4) 0%, 
                      rgba(216, 180, 254, 0.3) 50%, 
                      rgba(147, 51, 234, 0.4) 100%)`
                  : 'transparent',
                padding: isCurrentlyReading ? "12px 16px" : "6px 8px",
                borderRadius: "16px",
                border: isCurrentlyReading ? "2px solid rgba(216, 180, 254, 0.5)" : "none",
                position: 'relative',
                top: isCurrentlyReading ? "-2px" : "0px",
                display: 'inline-block',
                backdropFilter: isCurrentlyReading ? "blur(2px)" : "none",
                letterSpacing: "2px",
                filter: isCurrentlyReading 
                  ? "drop-shadow(0 0 15px rgba(147, 51, 234, 0.8))"
                  : "none",
              }}>
                {token.text}
              </span>

              {/* 魔法粒子效果 */}
              {isCurrentlyReading && Array.from({length: 12}).map((_, i) => (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    width: '4px',
                    height: '4px',
                    background: `hsl(${280 + i * 20}, 80%, 70%)`,
                    borderRadius: '50%',
                    left: Math.sin((frame + i * 30) * 0.04) * 80 + 50 + '%',
                    top: Math.cos((frame + i * 30) * 0.04) * 60 + 50 + '%',
                    opacity: 0.6 + Math.sin(frame * 0.05 + i) * 0.4,
                    boxShadow: `0 0 12px hsl(${280 + i * 20}, 80%, 70%)`,
                    pointerEvents: 'none',
                    transform: `scale(${0.5 + Math.sin(frame * 0.08 + i) * 0.5})`,
                  }}
                />
              ))}

              {/* 魔法能量波动 */}
              {isCurrentlyReading && (
                <div style={{
                  position: 'absolute',
                  inset: -15,
                  background: `radial-gradient(
                    ellipse, 
                    rgba(147, 51, 234, ${0.1 + Math.sin(frame * 0.06) * 0.1}), 
                    transparent 60%
                  )`,
                  borderRadius: '50%',
                  filter: 'blur(8px)',
                  pointerEvents: 'none',
                  animation: 'magical-pulse 3s ease-in-out infinite',
                }} />
              )}
            </div>
          );
        })}
      </div>
      
      <style jsx>{`
        @keyframes magical-pulse {
          0%, 100% { transform: scale(1); opacity: 0.3; }
          50% { transform: scale(1.2); opacity: 0.1; }
        }
      `}</style>
    </AbsoluteFill>
  );
};
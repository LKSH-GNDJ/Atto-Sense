/* AttoSense logo components — inlined so no public/ file needed.
   Use <LogoMark size={n} dark /> or <LogoFull dark /> anywhere.     */

export function LogoMark({ size = 40, dark = false }) {
  const ink      = dark ? '#E8E2D4' : '#1B1710'
  const sienna   = dark ? '#C8612A' : '#9B3D12'
  const fillOp   = dark ? '0.07'    : '0.04'
  const dotOp    = dark ? '0.5'     : '0.55'

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 120 120"
      width={size}
      height={size}
      fill="none"
      aria-label="AttoSense"
    >
      <defs>
        <clipPath id="as-clip">
          <polygon points="60,16 104,60 60,104 16,60" />
        </clipPath>
      </defs>

      {/* Outer diamond */}
      <polygon
        points="60,12 108,60 60,108 12,60"
        stroke={ink}
        strokeWidth="3"
        fill="none"
        strokeLinejoin="miter"
      />

      {/* Inner diamond — depth */}
      <polygon
        points="60,26 94,60 60,94 26,60"
        stroke={ink}
        strokeWidth="0.9"
        strokeOpacity="0.14"
        fill="none"
        strokeLinejoin="miter"
      />

      {/* Noisy input signal — left of diamond */}
      <path
        d="M 0,60 L 14,60 C 16,60 17,57 19,54 C 21,51 22,69 24,66 C 26,63 27,57 29,54 C 31,51 32,60 34,60 L 44,60"
        stroke={sienna}
        strokeWidth="1.5"
        strokeOpacity="0.4"
        fill="none"
        strokeLinecap="round"
      />

      {/* Clean classified pulse inside diamond + warm tint */}
      <g clipPath="url(#as-clip)">
        <polygon
          points="60,12 108,60 60,108 12,60"
          fill={sienna}
          fillOpacity={fillOp}
        />
        <path
          d="M 12,60 L 46,60 C 48,60 49,60 50,60 C 52,60 53,42 60,38 C 67,42 68,60 70,60 L 108,60"
          stroke={sienna}
          strokeWidth="2.2"
          fill="none"
          strokeLinecap="round"
        />
      </g>

      {/* Clean resolved output — right of diamond */}
      <path
        d="M 76,60 L 120,60"
        stroke={sienna}
        strokeWidth="1.5"
        strokeOpacity="0.4"
        fill="none"
        strokeLinecap="round"
      />

      {/* Centre dot — the classified intent */}
      <circle cx="60" cy="60" r="3.5" fill={sienna} />

      {/* Corner precision markers */}
      <circle cx="60"  cy="12"  r="2.5" fill={ink} fillOpacity={dotOp} />
      <circle cx="108" cy="60"  r="2.5" fill={ink} fillOpacity={dotOp} />
      <circle cx="60"  cy="108" r="2.5" fill={ink} fillOpacity={dotOp} />
      <circle cx="12"  cy="60"  r="2.5" fill={ink} fillOpacity={dotOp} />
    </svg>
  )
}

export function LogoFull({ dark = false, size = 'md' }) {
  const ink      = dark ? '#E8E2D4' : '#1B1710'
  const sienna   = dark ? '#C8612A' : '#9B3D12'
  const muted    = dark ? '#4A4638' : '#6C6656'
  const markSize = size === 'sm' ? 28 : size === 'lg' ? 52 : 38

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: markSize * 0.45 }}>
      <LogoMark size={markSize} dark={dark} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <div style={{
          fontFamily: "'Libre Baskerville', 'Liberation Serif', Georgia, serif",
          fontSize:   markSize * 0.72,
          fontWeight: 700,
          lineHeight: 1,
          letterSpacing: '-0.02em',
        }}>
          <span style={{ color: ink }}>Atto</span>
          <span style={{ color: sienna }}>Sense</span>
        </div>
        <div style={{
          fontFamily: "'Libre Baskerville', 'Liberation Serif', Georgia, serif",
          fontSize:   markSize * 0.27,
          fontStyle:  'italic',
          color:      muted,
          letterSpacing: '0.03em',
        }}>
          Intent Intelligence
        </div>
      </div>
    </div>
  )
}

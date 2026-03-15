"use client";

import { useEffect, useState } from "react";

export function AdvancedNebula() {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) return null;

  return (
    <div className="nebula-system" aria-hidden="true">
      {/* 1. SVG Definitions for Texture and Noise */}
      <svg className="nebula-svg" style={{ width: 0, height: 0, position: "absolute" }} aria-hidden="true">
        <defs>
          {/* Fractal noise for displacement (creates the smoky/fluid look) */}
          <filter id="aurora-filter">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.015"
              numOctaves="3"
              seed="5"
              result="noise"
            />
            {/* The scale determines how warped it gets */}
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="50" xChannelSelector="R" yChannelSelector="G" />
          </filter>

          {/* High-frequency static noise (film grain overlay) */}
          <filter id="grain-filter">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.8"
              numOctaves="3"
              stitchTiles="stitch"
              result="noiseOut"
            />
            <feColorMatrix type="saturate" values="0" in="noiseOut" result="noiseDesat" />
            <feComponentTransfer in="noiseDesat" result="noiseAlpha">
              <feFuncA type="linear" slope="0.5" />
            </feComponentTransfer>
            <feBlend mode="multiply" in="SourceGraphic" in2="noiseAlpha" />
          </filter>
        </defs>
      </svg>

      {/* 2. The Fluid Blobs Container (Displacement applied here) */}
      <div className="nebula-canvas">
        {/* Layered colored light blobs with smooth CSS motion */}
        <div className="nebula-blob nebula-blob--coral" />
        <div className="nebula-blob nebula-blob--amber" />
        <div className="nebula-blob nebula-blob--deep" />
        <div className="nebula-blob nebula-blob--highlight" />
      </div>

      {/* 3. The Grain Texture Overlay */}
      <div className="nebula-grain" />
    </div>
  );
}

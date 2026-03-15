"use client";



export function ModelLoader() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "60px 20px",
        gap: 24,
        background: "rgba(0,0,0,0.2)",
        borderRadius: 20,
        border: "1px dashed var(--border)",
      }}
    >
      <style>{`
        @keyframes pulseAura {
          0%, 100% { transform: scale(1); opacity: 0.15; }
          50% { transform: scale(1.4); opacity: 0; }
        }
        @keyframes pulseJoint {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.2); }
        }
        @keyframes flexArm {
          0%, 100% { transform: rotate(10deg); }
          50% { transform: rotate(-110deg); }
        }
      `}</style>
      <div style={{ position: "relative", width: 80, height: 80 }}>
        {/* Pulsing Background Aura */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            background: "var(--accent)",
            filter: "blur(8px)",
            animation: "pulseAura 1.5s ease-in-out infinite"
          }}
        />

        {/* Top Joint (Shoulder/Hip) */}
        <div 
          style={{ 
            position: "absolute", 
            left: "calc(50% - 6px)", 
            top: 16, 
            width: 12, 
            height: 12, 
            borderRadius: "50%", 
            background: "var(--accent-2)", 
            zIndex: 2,
            boxShadow: "0 0 10px var(--accent-2)"
          }} 
        />
        
        {/* Top Bone (Bicep/Thigh) */}
        <div 
          style={{ 
            position: "absolute", 
            left: "calc(50% - 3px)", 
            top: 22, 
            width: 6, 
            height: 30, 
            background: "rgba(255,255,255,0.8)", 
            borderRadius: 3 
          }} 
        />
        
        {/* Middle Joint (Elbow/Knee) */}
        <div
          style={{ 
            position: "absolute", 
            left: "calc(50% - 6px)", 
            top: 48, 
            width: 12, 
            height: 12, 
            borderRadius: "50%", 
            background: "var(--accent)", 
            zIndex: 2, 
            boxShadow: "0 0 15px var(--accent)",
            animation: "pulseJoint 1.5s ease-in-out infinite"
          }} 
        />
        
        {/* Bottom Bone & Joint Group (Forearm/Calf) */}
        <div
          style={{
            position: "absolute",
            left: "calc(50% - 3px)",
            top: 54,
            width: 6,
            height: 30,
            background: "rgba(255,255,255,0.8)",
            borderRadius: 3,
            transformOrigin: "top center",
            animation: "flexArm 1.5s ease-in-out infinite"
          }}
        >
          {/* Bottom Joint (Wrist/Ankle) */}
          <div 
            style={{ 
              position: "absolute", 
              left: -3, 
              bottom: -4, 
              width: 12, 
              height: 12, 
              borderRadius: "50%", 
              background: "var(--accent-2)", 
              zIndex: 2,
              boxShadow: "0 0 10px var(--accent-2)"
            }} 
          />
        </div>
      </div>

      <div style={{ textAlign: "center", display: "grid", gap: 8 }}>
        <h3 className="h3">Setting up your workout</h3>
        <p className="muted" style={{ maxWidth: 280, margin: "0 auto", fontSize: 13, lineHeight: 1.5 }}>
          Gearing up the local AI coach. Your camera feed stays entirely on your device for total privacy.
        </p>
      </div>
    </div>
  );
}

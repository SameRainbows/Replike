"use client";

import { useState, useEffect } from "react";

/**
 * Returns true if the device is a touch/mobile device.
 * Uses `pointer: coarse` media query — the most reliable signal for touch screens.
 * Falls back to `max-width: 768px` for SSR safety.
 */
export function useIsMobile(): boolean {
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const mq = window.matchMedia("(pointer: coarse), (max-width: 768px)");
        setIsMobile(mq.matches);

        const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
        mq.addEventListener("change", handler);
        return () => mq.removeEventListener("change", handler);
    }, []);

    return isMobile;
}

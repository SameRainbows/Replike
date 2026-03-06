"use client";

// Basic singleton pattern for Web Speech API
class SpeechEngine {
    private enabled: boolean = true;
    private lastSpoken: string = "";
    private lastSpokenTime: number = 0;

    public setEnabled(val: boolean) {
        this.enabled = val;
    }

    public isEnabled() {
        return this.enabled;
    }

    public speak(text: string, force: boolean = false) {
        if (!this.enabled || typeof window === "undefined" || !("speechSynthesis" in window)) return;

        // Prevent spamming the exact same phrase multiple times per second
        const now = Date.now();
        if (!force && text === this.lastSpoken && now - this.lastSpokenTime < 2000) {
            return;
        }

        // Cancel any currently speaking phrase so it doesn't queue up
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        // Try to find a good English voice
        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find(v => v.lang.startsWith("en-") && v.name.includes("Google"))
            || voices.find(v => v.lang.startsWith("en-"));
        if (preferred) {
            utterance.voice = preferred;
        }

        utterance.rate = 1.1; // Slightly faster for workout contexts
        utterance.pitch = 1.0;

        window.speechSynthesis.speak(utterance);

        this.lastSpoken = text;
        this.lastSpokenTime = now;
    }
}

export const speechCoach = new SpeechEngine();

export function updateVoiceSettings(enabled: boolean) {
    speechCoach.setEnabled(enabled);
}

// Ensure voices are loaded (they load async in some browsers)
if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {
        // Just trigger the load
        window.speechSynthesis.getVoices();
    };
}

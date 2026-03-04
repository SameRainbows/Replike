import { useState, useCallback } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { ExerciseId, Calibration } from "../exercises/types";
import { dist, angleDeg, getLandmark, isLandmarkConfident } from "../exercises/utils";

export function useCalibration(
    calibrationEnabled: boolean,
    calibration: Calibration,
    setCalibration: React.Dispatch<React.SetStateAction<Calibration>>
) {
    const [autoCalib, setAutoCalib] = useState<{ active: boolean; stableMs: number }>({ active: false, stableMs: 0 });
    const [autoCalibHint, setAutoCalibHint] = useState("");
    const [manualCalibOpen, setManualCalibOpen] = useState(false);
    const [manualCalibStep, setManualCalibStep] = useState(0);

    const [calibCache, setCalibCache] = useState<any>({});

    const isCalibrated = useCallback(
        (exercise: ExerciseId) => {
            if (!calibrationEnabled) return true;
            if (exercise === "jumping_jacks") return !!calibration.jumping_jacks;
            if (exercise === "squats") return !!calibration.squats;
            if (exercise === "lunges") return !!calibration.lunges;
            if (exercise === "high_knees") return !!calibration.high_knees;
            if (exercise === "jump_squats") return !!calibration.jump_squats;
            if (exercise === "burpees" || exercise === "push_ups" || exercise === "sit_ups" || exercise === "plank") return true;
            return true;
        },
        [calibrationEnabled, calibration]
    );

    const captureCalibrationFrame = useCallback(
        (exercise: ExerciseId, step: number, landmarks: NormalizedLandmark[]) => {
            if (exercise === "jumping_jacks") {
                const lShoulder = getLandmark(landmarks, 11);
                const rShoulder = getLandmark(landmarks, 12);
                const lAnkle = getLandmark(landmarks, 27);
                const rAnkle = getLandmark(landmarks, 28);
                const lWrist = getLandmark(landmarks, 15);
                const rWrist = getLandmark(landmarks, 16);
                const nose = getLandmark(landmarks, 0);

                if (!lShoulder || !rShoulder || !lAnkle || !rAnkle || !lWrist || !rWrist || !nose) return;

                const sw = dist(lShoulder, rShoulder);
                const aw = dist(lAnkle, rAnkle);
                const aRatio = aw / Math.max(sw, 1e-6);
                const writsY = (lWrist.y + rWrist.y) / 2;
                const armsLift = nose.y - writsY;

                if (step === 0) {
                    setCalibCache((c: any) => ({ ...c, jjClosedRatio: aRatio, jjClosedLift: armsLift }));
                } else {
                    setCalibration((c) => ({
                        ...c,
                        jumping_jacks: {
                            closedAnkleRatio: calibCache.jjClosedRatio ?? 1.1,
                            closedArmsLift: calibCache.jjClosedLift ?? 0,
                            openAnkleRatio: aRatio,
                            openArmsLift: armsLift,
                        },
                    }));
                }
            } else if (exercise === "squats" || exercise === "jump_squats" || exercise === "lunges") {
                const lHip = getLandmark(landmarks, 23);
                const rHip = getLandmark(landmarks, 24);
                const lKnee = getLandmark(landmarks, 25);
                const rKnee = getLandmark(landmarks, 26);
                const lAnkle = getLandmark(landmarks, 27);
                const rAnkle = getLandmark(landmarks, 28);

                if (!lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) return;

                const lKneeAngle = angleDeg(lHip, lKnee, lAnkle);
                const rKneeAngle = angleDeg(rHip, rKnee, rAnkle);
                const activeAngle = Math.min(lKneeAngle, rKneeAngle);

                if (step === 0) {
                    setCalibCache((c: any) => ({ ...c, topKnee: activeAngle }));
                } else {
                    setCalibration((c) => ({
                        ...c,
                        [exercise]: {
                            topKneeAngle: calibCache.topKnee ?? 175,
                            bottomKneeAngle: activeAngle,
                        },
                    }));
                }
            } else if (exercise === "high_knees") {
                const lHip = getLandmark(landmarks, 23);
                const rHip = getLandmark(landmarks, 24);
                const lKnee = getLandmark(landmarks, 25);
                const rKnee = getLandmark(landmarks, 26);

                if (!lHip || !rHip || !lKnee || !rKnee) return;

                const hipY = (lHip.y + rHip.y) / 2;
                const leftLift = hipY - lKnee.y;
                const rightLift = hipY - rKnee.y;
                const maxLift = Math.max(leftLift, rightLift);

                if (step === 0) {
                    setCalibCache((c: any) => ({ ...c, downLift: maxLift }));
                } else {
                    setCalibration((c) => ({
                        ...c,
                        high_knees: {
                            downLift: calibCache.downLift ?? -0.05,
                            upLift: maxLift,
                        },
                    }));
                }
            }
        },
        [calibCache, setCalibration]
    );

    return {
        autoCalib,
        setAutoCalib,
        autoCalibHint,
        setAutoCalibHint,
        manualCalibOpen,
        setManualCalibOpen,
        manualCalibStep,
        setManualCalibStep,
        calibCache,
        setCalibCache,
        isCalibrated,
        captureCalibrationFrame,
    };
}

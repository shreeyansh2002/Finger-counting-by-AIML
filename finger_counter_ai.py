"""
finger_counter_ai.py
Real-time hand gesture recognition combining:
  - the original rule-based convexity-defect finger counter
  - a trained ML classifier (RandomForest) running on shape features

Both predictions are shown side by side so the two approaches can be
compared directly -- this comparison is the main "result" of the project.
"""

import cv2
import numpy as np
import joblib
from features import extract_features

MODEL_FILE = "hand_gesture_model.pkl"


def get_hand_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask_hsv = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
    mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def rule_based_count(contour, hull_idx):
    if hull_idx is None or len(hull_idx) <= 3:
        return 0
    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return 0
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start, end, far = contour[s][0], contour[e][0], contour[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b * c == 0:
            continue
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        if angle <= np.radians(90) and d > 10000:
            count += 1
    return min(count + 1, 5)


def main():
    model = joblib.load(MODEL_FILE)
    cap = cv2.VideoCapture(0)
    print("AI gesture recognition started. Press Q to quit.")

    history = []          # smoothing buffer so predictions don't flicker
    HISTORY_LEN = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        roi_top, roi_bottom = 80, h - 80
        roi_left, roi_right = 80, w - 200
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        mask = get_hand_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rule_count = 0
        ml_label, ml_conf = "-", 0.0

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 5000:
                cv2.drawContours(frame, [contour + [roi_left, roi_top]], -1, (0, 200, 100), 2)

                hull_idx = cv2.convexHull(contour, returnPoints=False)
                rule_count = rule_based_count(contour, hull_idx)

                feats = extract_features(contour, roi.shape).reshape(1, -1)
                probs = model.predict_proba(feats)[0]
                pred_idx = int(np.argmax(probs))
                ml_label = model.classes_[pred_idx]
                ml_conf = probs[pred_idx]

                history.append(ml_label)
                if len(history) > HISTORY_LEN:
                    history.pop(0)

        smoothed = max(set(history), key=history.count) if history else "-"

        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (w, 90), (80, 80, 80), -1)
        cv2.putText(frame, f"Rule-based count: {rule_count}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"ML prediction: {smoothed}  (conf {ml_conf * 100:.0f}%)",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow("AI Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

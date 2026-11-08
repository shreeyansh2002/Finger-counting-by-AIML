"""
collect_data.py
Interactive tool to build a labeled dataset of hand-gesture feature
vectors for training the classifier in train_model.py.

Controls:
  0-5   -> label the current frame as that many fingers and save it
  q     -> quit

Aim for at least 40-50 samples per class (0 to 5), varying your hand's
distance, angle, and lighting between samples -- a model trained on
near-identical samples will not generalize.
"""

import cv2
import csvgit inioo
import os
import numpy as np
from features import extract_features, FEATURE_NAMES

DATA_FILE = "hand_data.csv"


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


def main():
    file_exists = os.path.isfile(DATA_FILE)
    f = open(DATA_FILE, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(FEATURE_NAMES + ["label"])

    cap = cv2.VideoCapture(0)
    print("Show a gesture, then press 0-5 to label and save it.")
    print("Collect 40-50+ samples per class. Press Q to quit.")

    saved_count = 0

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

        current_contour = None
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) > 5000:
                current_contour = biggest
                cv2.drawContours(frame, [biggest + [roi_left, roi_top]], -1, (0, 200, 100), 2)

        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (w, 60), (80, 80, 80), -1)
        cv2.putText(frame, f"Saved: {saved_count}  |  Press 0-5 to label, Q to quit",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Data collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key != 255 and chr(key) in "012345":
            if current_contour is not None:
                feats = extract_features(current_contour, roi.shape)
                writer.writerow(list(feats) + [chr(key)])
                f.flush()
                saved_count += 1
                print(f"Saved sample #{saved_count} -> label {chr(key)}")
            else:
                print("No hand detected, sample not saved.")

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved_count} samples written to {DATA_FILE}")


if __name__ == "__main__":
    main()

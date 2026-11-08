# AI-based hand gesture recognition

Adds a real supervised-learning layer on top of the original rule-based
finger counter, so the two approaches can be compared directly.

## Pipeline

1. `features.py` -- turns a hand contour into a numeric feature vector
   (solidity, aspect ratio, defect stats, 7 Hu moments = 14 features).
2. `collect_data.py` -- run this first. Show gestures to the webcam,
   press 0-5 to label and save each sample to `hand_data.csv`.
3. `train_model.py` -- trains a RandomForestClassifier on the CSV,
   prints accuracy / classification report / confusion matrix, saves
   `hand_gesture_model.pkl`.
4. `finger_counter_ai.py` -- live demo. Shows the rule-based count and
   the ML prediction side by side on the same feed.

## Setup

```
pip install -r requirements.txt
python collect_data.py        # collect 40-50+ samples per class (0-5)
python train_model.py         # train and evaluate
python finger_counter_ai.py   # run the live hybrid demo
```

## Why this counts as an AI project

The original script was 100% hand-tuned geometry (fixed HSV ranges, a
fixed angle threshold, a fixed depth threshold) -- no part of it was
learned from data. This version adds:

- **Feature engineering**: raw contours are converted into a structured
  feature vector rather than fed to fixed if/else rules.
- **Supervised learning**: a RandomForestClassifier is trained on
  labeled examples you collect yourself.
- **Train/test evaluation**: accuracy, a classification report, and a
  confusion matrix are computed on held-out data, not just eyeballed.
- **Feature importance analysis**: `train_model.py` prints which
  features the model actually relies on -- good evidence for a viva.

## Report/synopsis structure

1. Problem statement and objectives
2. Related work (classical CV vs. landmark-based methods like MediaPipe;
   note why classical CV was chosen -- e.g. MediaPipe/Python version
   incompatibility -- if relevant to your setup)
3. System architecture (the pipeline diagram from earlier in this chat)
4. Methodology: feature extraction, model choice (RandomForest -- robust
   on small tabular datasets, interpretable via feature importances),
   train/test split
5. Results: accuracy, confusion matrix, rule-based vs. ML comparison
   (where do they agree/disagree, and why)
6. Limitations: lighting/skin-tone sensitivity of the skin-color mask,
   small dataset size, single-user training data
7. Future scope: larger/more diverse dataset, deep learning (CNN on raw
   ROI or MediaPipe landmarks) as a natural next step
8. References

## Notes

- Collect data with a few different distances/angles/lighting
  conditions per class -- a model trained on near-identical frames
  will not generalize past your exact setup.
- If accuracy is low, check `train_model.py`'s feature importance
  output -- it tells you which features aren't earning their keep.

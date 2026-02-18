# MLB Strikeout Prediction

## Objective
Predict pitcher-game strikeouts using MLB Statcast pitch-level data.
Evaluate regression performance and frame predictions into Over/Under decisions.
Compare traditional ML, deep learning, and Bayesian uncertainty approaches.

## Models
- Mean baseline
- Linear regression
- XGBoost
- Feed-forward Neural Network
- Bayesian Neural Network

## Pipeline Stages
1. Data acquisition (Statcast pitch-level data)
2. Data cleaning and preprocessing
3. Label engineering (strikeouts per pitcher-game)
4. Feature engineering (pitch → game aggregation)
5. Time-aware train/test split
6. Model training and evaluation
7. Over/Under framing
8. SHAP explainability
9. Uncertainty evaluation (BNN)

## Reproducibility
1. Install requirements
2. Run data_pull.py
3. Run cleaning.py
4. Run labels.py
5. Run features.py
6. Run training scripts

## Decision Framing
Regression predictions converted into Over/Under classification.

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

## Daily automated run (GitHub Actions)

The whole daily workflow runs unattended via `run_daily.py`, scheduled by
[.github/workflows/daily.yml](.github/workflows/daily.yml).

**What it does each day:** settles yesterday's flagged plays against MLB
boxscores, predicts *today's* slate (real DraftKings/FanDuel odds + confirmed
lineups + short-outing correction), cross-checks each play against oddsindex.com,
and posts a compact summary to Discord.

**Run it manually (local):**
```
python run_daily.py                 # targets today (US Eastern)
RUN_DATE=2026-08-10 python run_daily.py   # a specific date
```
Needs `ODDS_API_KEY` (in `.env` or env) for real odds, and
`DISCORD_WEBHOOK_URL` for the notification — both optional; the run degrades
gracefully (assumed lines / skipped notification) without them.

**Schedule:** cron `30 20 * * *` = **20:30 UTC = 9:30 PM UK**. The MLB season is
entirely in BST, so this is 9:30 PM UK every day it matters. To change it, edit
the `cron:` line (UTC; GitHub does not adjust for BST/GMT). You can also trigger a
run any time from the repo's **Actions → daily-strikeout-run → Run workflow**.

**Secrets** (repo → Settings → Secrets and variables → Actions):
`ODDS_API_KEY`, `DISCORD_WEBHOOK_URL`. Never commit these.

**When it fails:** GitHub emails you (the job exits non-zero on failure). Check
**Actions → the failed run → logs**; the predictions/ledger are also uploaded as
a run artifact. Note GitHub disables scheduled workflows after ~60 days with no
commits, and scheduled runs can fire a little late during busy periods.

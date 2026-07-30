# Video Recommendation System

A recommendation engine that predicts which video a user is likely to watch next, based on their
watch history — similar to how TikTok or YouTube personalize a feed. Built on a simulated dataset of
5 users, 15 videos, and their watch interactions.

## How it works

For each user, features are built from their watch history: how much they've liked a video's category
before, whether the video's language matches theirs, video duration, and time of day. A **logistic
regression model** is trained on these features to predict the probability a user will like a given
video. Unwatched videos are ranked by that probability, and the top matches are recommended.

New users with no watch history get shown popular videos in their own language instead of a
personalized guess, since there's no data yet to base one on.

## Model

- **Algorithm:** Logistic Regression (scikit-learn)
- **Features:** category affinity (smoothed with Bayesian shrinkage), language match, video duration,
  time of day
- **Evaluation:** Leave-one-out cross-validation
- **Baseline comparison:** outperforms a naive "recommend most popular" baseline

## Stack

Python · pandas · scikit-learn · SQLite · Jupyter


## Running it

```bash
pip3 install -r requirements.txt
jupyter notebook
```

Open `notebook/ml_recommender.ipynb` and run top to bottom.

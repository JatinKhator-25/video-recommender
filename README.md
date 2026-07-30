# Video Recommendation System

A recommender that predicts which video a user should watch next — similar to how TikTok decides
what shows up on your feed. Built on a simulated dataset (5 users, 15 videos) to work through a real
recommender pipeline end to end: feature engineering, a trained model, proper evaluation, context, and
cold-start handling.

## The idea

For every user, we describe their taste with a few honest signals: how much they've liked this video's
*category* before (based on other videos, never the one being predicted), whether the video's language
matches theirs, how long the video is, and whether it's night or day.

A **logistic regression model**, trained on the user's real interaction history, takes those signals and
predicts the probability they'll like a given video. Unwatched videos are ranked by that probability,
and the top few become the recommendation.

New users with no watch history skip the model entirely and get shown popular videos in their language —
there's nothing to learn from yet, so a personalized guess would just be noise.

## Why a trained model instead of a fixed formula

The first version of this project scored videos with hand-picked weights (category mattered most,
language second, duration least). That works, but it's not really learning anything — those numbers
came from intuition, not data.

Swapping in logistic regression means the model derives those weights itself from actual behavior. It's
a small dataset, but it's the real mechanism: given examples of what a user did and didn't like, the
model learns which signals actually predict that outcome, rather than being told the answer in advance.

## Two real bugs, found and fixed along the way

**Language dominating category.** An early version combined all features into one similarity score, and
a user's strong language signal quietly outweighed their category preference — a comedy-and-gaming fan
was getting music and cooking recommended just because the language matched. Fixed by scoring each
feature group separately and weighting them explicitly.

**Untested categories beating disliked ones.** When a user had never watched a category, it defaulted to
a neutral score — which meant totally untested content could outrank a category the user had shown real
(if imperfect) interest in. Fixed with Bayesian smoothing: blending a user's observed preference for a
category with the platform-wide average, weighted by how much data actually supports it. Same idea
IMDb uses to keep a movie with 3 five-star ratings from outranking one with 10,000.

## Does it actually work?

Evaluated with leave-one-out cross-validation — for every interaction, the model is trained on all the
*others* and tested on the one held out, so the accuracy reflects genuine prediction, not memorization.

This was also compared against a naive baseline (just recommend whatever's most popular overall), which
scored 0% — confirming the personalization is contributing something real, not just added complexity.

## Stack

Python, pandas, scikit-learn (LogisticRegression), SQLite, Jupyter.

## What I'd change at real scale

- Add collaborative filtering once there's enough interaction density for it to help, and combine it
  with this model
- Replace hand-built features with learned embeddings
- Proper time-based train/test split instead of leave-one-out on a small sample
- Wrap it in an API instead of a notebook, and A/B test the time-of-day rules against real engagement

## Project structure

```
video-recommender/
├── data/                     recommender.db (SQLite: users, videos, interactions)
├── notebook/
│   ├── recommender_pipeline.ipynb   first version — similarity scoring with hand-picked weights
│   └── ml_recommender.ipynb         current version — trained logistic regression model
├── assets/                   diagrams and result charts used in this README
├── requirements.txt
└── README.md
```

`ml_recommender.ipynb` is the version to look at first — it's what the sections above describe.
`recommender_pipeline.ipynb` is kept as the earlier iteration, useful for seeing how the approach
evolved from a hand-tuned formula to a trained model.

## Running it

```bash
pip3 install -r requirements.txt
jupyter notebook
```

Open `notebook/ml_recommender.ipynb` and run the cells in order — it loads the database, builds
features, trains the model, evaluates it, and generates recommendations for each user.

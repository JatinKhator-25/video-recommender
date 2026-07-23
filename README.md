# Video Recommendation System

A content-based video recommender (TikTok-style) built to demonstrate practical understanding of
recommendation system fundamentals: feature engineering, similarity scoring, contextual re-ranking,
cold-start handling, and model evaluation against a baseline.

## Overview

This project simulates a small video platform — 5 users, 15 videos, and a set of watch interactions
(watch time, completion rate, likes, timestamps) — and builds a recommender that predicts which
unwatched videos a user is most likely to enjoy next.

The dataset is intentionally small. The goal wasn't to build something production-scale; it was to
implement every core piece of a real recommender system end-to-end, understand why each piece works,
and be able to defend every design decision — including the ones that didn't work on the first try.

## Architecture

```
SQLite DB (users, videos, interactions)
        │
        ▼
Feature Engineering
  - Video vectors: one-hot category + language, normalized duration
  - User vectors: weighted average of watched video vectors
    (weight = completion_rate + likedbonus)
        │
        ▼
Similarity Scoring
  - Per-feature-group cosine similarity (category, language, duration)
  - Combined with explicit weights (0.75 / 0.15 / 0.10)
        │
        ▼
Contextual Re-ranking
  - Time-of-day boost (night categories vs day categories)
        │
        ▼
Cold-Start Fallback
  - New users → popularity ranking within their language
        │
        ▼
Top-N Recommendations
```

## Key Design Decisions

### Why content-based filtering, not collaborative filtering
With only 5 users and 15 videos, a user-item interaction matrix is far too sparse for collaborative
filtering to find meaningful patterns. Content-based filtering (matching user taste profiles to video
attributes) produces a meaningful signal even at this scale. At production scale with thousands of
users, a hybrid approach (content-based + matrix factorization) would outperform either alone.

### Why per-feature-group similarity instead of one concatenated cosine similarity
The first version concatenated all one-hot features (category + language) into a single vector and
computed one cosine similarity. This caused a real bug: users with preferences concentrated in a single
strong dimension (e.g. one dominant language) got recommendations driven by that dimension, even when
category was the more meaningful signal. A comedy-and-gaming fan was getting cooking and music videos
recommended purely because they matched on language.

**Fix:** compute cosine similarity separately within each feature group (category, language, duration),
then combine with explicit weights — category weighted highest (0.75), since content type matters more
to taste than language, followed by language (0.15) and duration (0.10).

### Why the user profile is a weighted average, not a simple average
A video watched to 95% completion and liked should count far more toward a user's taste profile than
one watched for 10 seconds and abandoned. The weight formula used is:

```
weight = completion_rate + (liked * 0.5)
```

This is a simple, fully explainable scoring choice rather than a black-box weighting.

### Cold-start handling
A brand-new user with no watch history has no profile vector to compare against video vectors. Instead
of failing or guessing randomly, new users are shown the most popular videos within their stated
language — a common, defensible fallback used across real platforms until enough behavioral data
accumulates.

### Contextual re-ranking
Time-of-day is treated as a request-time adjustment on top of the static similarity score, not baked
into the user's profile vector. Content preference (what someone generally likes) is relatively stable;
context (when they're watching) is dynamic and should influence ranking per-request rather than be
learned as a permanent trait.

## Evaluation

**Method:** Leave-one-out validation. For each user, their most recent liked video was held out, the
profile was rebuilt without it, and the system checked whether it still ranked that video in the top 3
recommendations out of the remaining candidates.

**Result:**

| Model | Precision@3 |
|---|---|
| Popularity baseline (recommend whatever's most-liked overall) | 0% |
| Content-based recommender (this project) | 40% |

**Honest limitation:** with only 5 users, this is a directional signal, not a statistically robust
evaluation. The misses were traced individually and all clustered around users with preferences split
across multiple categories with no single dominant signal — a known limitation of content-based
similarity on sparse, diffuse data, rather than a flaw in the scoring logic itself. At production scale,
with more interactions per user, a proper time-based train/test split with rolling Precision@K and
Recall@K tracking would be used instead.

## What I'd do differently at production scale

- Add a collaborative filtering component (matrix factorization / SVD) once there's enough interaction
  density to support it, and combine it with the content-based score in a hybrid model
- Replace one-hot tags with learned embeddings to capture nuance beyond exact category/language matches
- A/B test the contextual re-ranking rules against real engagement data instead of hand-set boosts
- Track Precision@K and Recall@K over rolling time windows on a much larger, real interaction dataset
- Move from a notebook pipeline to a proper API (FastAPI) with a lightweight demo UI

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (cosine similarity)
- SQLite (relational schema: users, videos, interactions with foreign keys)
- Jupyter Notebook
- matplotlib (EDA visualization)

## Project Structure

```
video-recommender/
├── data/
│   └── recommender.db
├── notebooks/
│   ├── 01_setup_db.ipynb
│   └── 02_recommender.ipynb
├── requirements.txt
└── README.md
```

## Setup

```bash
pip3 install -r requirements.txt
jupyter notebook
```

Open the notebooks in order — `01_setup_db.ipynb` creates and populates the database,
then the recommender logic (feature engineering, similarity, context, cold-start, evaluation)
runs through the subsequent cells.

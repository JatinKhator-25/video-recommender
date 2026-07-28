# Video Recommendation System

A small content-based recommender that predicts which video a user should watch next — similar to how TikTok decides what shows up on your feed. Built on a simulated dataset (5 users, 15 videos) to work through the full pipeline of a real recommender system: turning behaviour into features, scoring similarity, adjusting for context, and handling brand-new users.

## The idea

Every user's watch history gets turned into a "taste profile" — basically a set of numbers describing how much they lean toward each video category (comedy, gaming, sports, etc.) and language, weighted so videos they actually finished and liked count more than ones they skipped. Every video gets a similar profile based on its own attributes.

Recommending a video is then just checking how closely a user's taste profile lines up with a video's profile, using cosine similarity — a way of measuring how "aligned" two sets of numbers are.

On top of that:
- Time of day shifts the ranking slightly (comedy/gaming get a boost at night, cooking/tech during the day)
- New users with no history get shown popular videos in their language instead of a personalized guess, since there's nothing to learn from yet

## Why it works

The core assumption is the same one every recommender system relies on: people who liked certain kinds of content before are likely to enjoy similar content again. The math is just a consistent way of applying that assumption across every user and video instead of guessing by hand.

The one part that wasn't obvious going in: combining all features (category + language) into a single similarity score caused language to quietly dominate the recommendations — a user who mostly watched English videos kept getting recommended totally unrelated categories, just because they matched on language. Fixed by scoring category, language, and duration similarity separately and combining them with different weights (category matters most, language second, duration least).

## Does it actually work?

Tested by hiding a video each user had genuinely liked, rebuilding their profile without it, and checking if the system could still figure out to recommend it back.

- Popularity-only baseline (just recommend whatever's most liked): 0%
- This model: 40%

Small sample, so take it as a directional result, not a hard statistic — but it's a clean signal that the personalization is doing something real, not just adding complexity for no reason.

## Stack

Python, pandas, scikit-learn, SQLite, Jupyter.

## What I'd change at real scale

- Add collaborative filtering once there's enough data density for it to be useful, and combine it with the content-based score
- Replace one-hot category/language features with learned embeddings
- Proper time-based train/test evaluation instead of leave-one-out on a handful of users
- Wrap it in an API instead of a notebook

## Running it

```bash
pip3 install -r requirements.txt
jupyter notebook
```

Run `notebook/recommender_pipeline.ipynb` — it builds the database, generates the features, and walks through scoring, evaluation, and the fixes made along the way.
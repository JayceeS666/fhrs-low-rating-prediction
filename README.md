# FHRS Low-Rating Prediction

I built this to predict which food businesses in England are likely to get a low hygiene rating, so local authorities can prioritise inspections where they're actually needed. It's part of my MSc dissertation (AI and Government, University of Birmingham).

Quick context: FHRS stands for Food Hygiene Rating Scheme — it's the official UK system (run by the Food Standards Agency) that gives food businesses a 0-5 hygiene score. My question was: can I predict, before an inspection even happens, which businesses are likely to score low?

## What's in here

I pulled England's FHRS data (297 local authorities, last 5 years, ~400K businesses after cleaning), and tried to predict whether a business would score 0-3 ("high risk") using stuff you'd actually know *before* inspecting it — business type, location, how long since the last inspection, and how deprived the area is.

One thing worth flagging: the raw data also has three sub-scores (Hygiene, Structural, Confidence in Management) that get combined into the final rating. I left these out on purpose — using them would basically be letting the model peek at the answer, since they're generated *during* the inspection, not before it.

I tried three models, with SMOTE to handle the class imbalance (only ~9% of businesses are actually high-risk):

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.748 | 0.254 | 0.310 | 0.279 |
| Random Forest | 0.782 | 0.295 | 0.368 | 0.327 |
| **XGBoost** | **0.787** | 0.324 | 0.283 | 0.302 |

XGBoost came out on top, so I went with that as the final model. I also tuned the threshold (using Youden's J), which pushes recall up to 0.78 if you're willing to accept more false positives — see `outputs/confusion_matrices.png` for what that trade-off looks like.

I used SHAP to explain the predictions rather than just spit out a risk score. Turns out business type matters most (takeaways and restaurants are riskier than, say, schools), followed by region and how long it's been since the last check. Details in `outputs/shap_summary.png`.

## Repo layout

```
.
├── fhrs_pipeline.py       # the whole thing: load → clean → build features → train → SHAP → plots
├── england_lookup.py      # local authority name lookups (kept separate so the main file stays readable)
├── requirements.txt
├── data/
│   ├── FHRS_All_en-GB.csv.gz
│   └── File_10_IoD2025_Local_Authority_District_Summaries.xlsx
├── outputs/               # plots from a full run
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── shap_summary.png
├── docs/
│   └── DEVELOPMENT_LOG.md # how the coding scheme came together, decisions I tried and dropped
└── README.md
```

## Running it

I've already bundled the data in `data/`, so:

I ran this on Python 3.10 — didn't try anything newer, but heads up: really new Python versions (3.13+) sometimes trip up on packaging with some of these libraries, so I'd stick to 3.9-3.12 if you're setting up from scratch.

```bash
pip install -r requirements.txt
python fhrs_pipeline.py
```

## About the data

The FHRS snapshot in `data/` (28.6MB, gzip'd down from a 143MB raw export) is what I pulled from https://ratings.food.gov.uk/open-data on 16-17 August 2026 -- the latest rating in there is from 14 August 2026. I compressed and included it rather than just linking to it, because FHRS updates daily, and I wanted anyone running this to get the exact same numbers I did, not something that's drifted slightly.

The deprivation data (`File_10...xlsx`, 226KB) is from the 2025 English Indices of Deprivation: https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025

Pandas reads the `.gz` file directly, so no need to unzip anything.

If you want the full story of how I put the coding scheme together -- including the stuff that didn't work -- I wrote it all up in [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

## License

MIT, see [LICENSE](LICENSE).

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

I used SHAP to explain the predictions rather than just spit out a risk score. Turns out business type matters most (takeaways and restaurants are riskier than, say, schools), followed by region and how long it's been since the last check. Details in `outputs/shap_summary.png`.<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_980_1875)">
<rect width="16" height="16" fill="url(#paint0_radial_980_1875)"/>
<mask id="mask0_980_1875" style="mask-type:alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="16" height="16">
<rect width="16" height="16" fill="url(#paint1_radial_980_1875)"/>
<rect width="16" height="16" fill="url(#paint2_radial_980_1875)"/>
</mask>
<g mask="url(#mask0_980_1875)">
<g filter="url(#filter0_f_980_1875)">
<path d="M13.9999 14.3999C16.4867 11.9131 16.7466 6.74661 16.8799 4.71994L18.4799 7.75994L17.0399 19.6799H9.11994L1.43994 16.6399C4.71994 16.5599 11.6799 16.7199 13.9999 14.3999Z" fill="#00FCFD"/>
</g>
</g>
<g clip-path="url(#clip1_980_1875)">
<path d="M8.99583 9.67423C8.99583 9.67423 9.26277 9.55776 9.63207 9.42444C11.0365 8.93523 12.5247 8.62509 14.0697 8.52096C14.1155 8.51792 14.1286 8.45719 14.0882 8.43554C12.97 7.83407 11.7474 7.40187 10.4544 7.17324C10.9971 6.25751 11.6044 5.3847 12.2702 4.56115C12.295 4.53049 12.2718 4.48524 12.2324 4.48739C12.0719 4.49621 11.9106 4.50208 11.7485 4.50473C12.2445 3.6564 12.7958 2.84452 13.3975 2.07368C13.4251 2.03831 13.3906 1.98855 13.3479 2.00236C11.6917 2.53996 10.1218 3.26897 8.66443 4.16373C8.10763 4.50561 7.56709 4.87168 7.04438 5.26039C6.91292 5.35835 6.78234 5.45758 6.65303 5.55848C6.58525 5.44249 5.56138 3.74124 4.09365 4.21702C3.99501 4.24896 3.88951 4.25327 3.78959 4.22555C3.12611 4.04138 2.46685 3.84997 1.98283 3.70734C1.92513 3.6903 1.89369 3.77239 1.94805 3.79805C2.6037 4.10809 3.53323 4.61072 3.97943 5.14617C4.55406 5.85196 4.13793 6.71802 4.18211 7.51658C4.24255 8.60736 4.95824 9.45255 5.82097 10.0521C6.3091 10.3913 6.85463 10.6268 7.36432 10.931C8.68363 11.7184 9.87246 12.7451 10.7928 13.9798C10.8268 14.0255 10.898 13.9863 10.8774 13.9333C10.5029 12.9689 9.21388 10.1505 8.99553 9.67423H8.99583Z" fill="url(#paint3_linear_980_1875)"/>
</g>
</g>
<defs>
<filter id="filter0_f_980_1875" x="-0.960059" y="2.31994" width="21.84" height="19.76" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
<feFlood flood-opacity="0" result="BackgroundImageFix"/>
<feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"/>
<feGaussianBlur stdDeviation="1.2" result="effect1_foregroundBlur_980_1875"/>
</filter>
<radialGradient id="paint0_radial_980_1875" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1.92 1.6) rotate(45.5989) scale(21.6481 22.5405)">
<stop offset="0.599832" stop-color="#3F85FF"/>
<stop offset="0.822115" stop-color="#AAA3FF"/>
</radialGradient>
<radialGradient id="paint1_radial_980_1875" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(3.2) rotate(50.9061) scale(21.9899 22.8964)">
<stop offset="0.449706" stop-color="#3F85FF"/>
<stop offset="0.822115" stop-color="#AAA3FF"/>
</radialGradient>
<radialGradient id="paint2_radial_980_1875" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(16 18.32) rotate(-128.946) scale(7.50893 14.804)">
<stop offset="0.385471" stop-color="#00FCFD"/>
<stop offset="1" stop-color="#00FCFD" stop-opacity="0"/>
</radialGradient>
<linearGradient id="paint3_linear_980_1875" x1="11.4667" y1="14" x2="3.6" y2="4.13333" gradientUnits="userSpaceOnUse">
<stop stop-color="#E8EDF4"/>
<stop offset="0.445071" stop-color="#F1F6FA"/>
<stop offset="0.717298" stop-color="white"/>
</linearGradient>
<clipPath id="clip0_980_1875">
<rect width="16" height="16" rx="2.8125" fill="white"/>
</clipPath>
<clipPath id="clip1_980_1875">
<rect width="12.1923" height="12" fill="white" transform="translate(1.92017 2)"/>
</clipPath>
</defs>
</svg>


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
```<img width="16" height="16" alt="logo" src="https://github.com/user-attachments/assets/b94be167-5057-404d-bd6d-660ffa61ff94" />


## Running it

I've already bundled the data in `data/`, so:

```bash
pip install -r requirements.txt
python fhrs_pipeline.py
```

## About the data

The FHRS snapshot in `data/` (28.6MB, gzip'd down from a 143MB raw export) is what I pulled from https://ratings.food.gov.uk/open-data on 16–17 August 2026 — the latest rating in there is from 14 August 2026. I compressed and included it rather than just linking to it, because FHRS updates daily, and I wanted anyone running this to get the exact same numbers I did, not something that's drifted slightly.

The deprivation data (`File_10...xlsx`, 226KB) is from the 2025 English Indices of Deprivation: https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025

Pandas reads the `.gz` file directly, so no need to unzip anything.

If you want the full story of how I put the coding scheme together — including the stuff that didn't work — I wrote it all up in [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

## License

MIT, see [LICENSE](LICENSE).

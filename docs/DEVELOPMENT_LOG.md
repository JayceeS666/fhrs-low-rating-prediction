# How I built out the coding scheme

A log of the calls I made while putting `fhrs_pipeline.py` together — what I went with, what I tried and binned, and why. Writing it up so anyone who hasn't seen my working files can still follow what I did and check it if they want.

---

## 1. What counts as "high risk"

Went with: rating 0-3 = high risk, 4-5 = not.

Tried a stricter cutoff first (0-2 only), but that only gave about 3% positive cases — way too thin for the models to actually learn from. 0-3 keeps enough signal to work with and still means something regulation-wise. Ran this by my supervisor before touching any models.

## 2. Sorting out what's actually England

The raw FHRS file just dumps England, Wales, NI, and Scotland together — no country column at all. Had to use FSA's own regional groupings to build a list of English local authorities myself, then check it against what's actually in the data. Expected 298, got 297 matches — looked into the one that didn't show up (Newcastle upon Tyne), turned out it just had zero records in this file, not a naming mismatch.

That, plus only keeping `SchemeType == 'FHRS'` (drops Scotland's separate scheme), is the England filter I use everywhere.

## 3. Why the FHRS sub-scores didn't make it in

The raw data hands you `Hygiene`, `Structural`, and `ConfidenceInManagement` — three ready-to-go numeric columns. Didn't touch them.

Why: these three literally get combined into the final rating. Using them as features is basically handing the model the answer along with the question. And realistically, at the point you'd actually want a prediction, none of them exist yet — the inspection hasn't happened.

## 4. Chopping down the time window

Started with everything, back to 1998. My supervisor said old records probably don't say much about how things are now, so I cut it to the last 5 years before extraction. Just `RatingDate >= max(RatingDate) - 5 years`.

## 5. Duplicates

Same business name + same postcode twice = same place, listed twice. Keep whichever one's more recent, drop the rest.

## 6. Deprivation data — went through three versions

This is the one I flip-flopped on the most, so it's worth actually spelling out — the version I kept (the less precise one) isn't something you'd guess from just reading the code.

| Version | What it was | XGBoost AUC |
|---|---|---|
| v1 | No deprivation data | 0.774 |
| v2 (**went with this**) | IMD25 score by local authority, matched on name (had to hand-fix ~30 naming mismatches — "Kingston upon Hull, City of" vs "Hull City", stuff like that) | **0.787** |
| v3 | IMD25 at the much finer LSOA level — spatially matched each business's coordinates to ONS boundary polygons (worked for 77.9% of records, rest fell back to v2) | 0.776 |

Kept v2. Beat v3 on both tree-based models, even though it's the rougher option. Best guess: lat/long were already in the feature set in all three versions, so whatever extra detail LSOA data added, the model was probably already picking most of that up from the raw coordinates. And since the LSOA join only covered 77.9% of records, the gap likely added more noise than the precision was worth.

(v3 used `geopandas.sjoin` against an ONS boundary file — didn't bother including that here since it's not needed for the version I actually kept.)

## 7. Handling the class imbalance

Used SMOTE, training data only — bumps the high-risk class up to 50/50. Left the test set completely alone so the numbers I report still reflect the real imbalance.

Thought about just weighting the classes instead, but stuck with SMOTE to line up with Allen et al. (2019) and Oldroyd et al. (2021), my two closest comparisons.

## 8. One change that was just about getting it to run

First attempt at Random Forest was 300 trees, no depth limit. Basically never finished on ~550K rows after SMOTE. Had to kill it.

Capped it at depth 20, 150 trees — done in a few minutes, barely moved the numbers. Ran XGBoost with `tree_method='hist'` for the same reason, it's just faster at this scale.

Not a methodology call, just "this needed to actually finish" — mentioning it because the settings in `fhrs_pipeline.py` are the fixed version, not what I started with.

## 9. Picking a threshold

Used Youden's J (the ROC point that best balances catching true positives vs false alarms) alongside the usual 0.5 cutoff. Picked this specifically because Allen et al. (2019) and Oldroyd et al. (2021) both used it too, so my numbers actually line up with theirs.

## 10. A couple of mistakes I caught late

Going back through the dissertation against what the code actually outputs, found two things that didn't match:

- Said "298 local authorities" in one spot, "297" everywhere else — fixed to 297 throughout, since that's what's actually in the data (see #2).
- The sample size (402,815) and test-set size (100,704) I'd written down were from *before* the deprivation merge — 8 records got dropped there for not matching a local authority. Fixed to the re

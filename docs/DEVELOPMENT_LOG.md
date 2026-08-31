# How I built out the coding scheme

Basically a log of the decisions I made building `fhrs_pipeline.py` — what I went with, what I tried and dropped, and why. Writing it up so someone who's never seen my working files can follow how I coded things and check my process if they want to.

---

## 1. What counts as "high risk"

Went with: rating 0-3 = high risk, 4-5 = not.

Tried a stricter cutoff early on (0-2 only) but that only gave about 3% positive cases — too thin for the models to actually learn from. 0-3 keeps enough signal to work with and still means something in regulatory terms. Checked this with my supervisor before I started modelling anything.

## 2. Working out what's actually England

The raw FHRS file just dumps England, Wales, NI, and Scotland together — no country column, nothing. I used FSA's own regional groupings (East Counties, East Midlands, London, etc.) to build a list of English local authorities, then checked that against what's actually sitting in the data. Got 297 out of the ~298 I was expecting — looked into the one that didn't show up (Newcastle upon Tyne), turned out it just had zero records in this particular file, not a naming mismatch.

That plus only keeping `SchemeType == 'FHRS'` (drops Scotland's separate FHIS scheme) is the England filter I use everywhere.

## 3. Why the FHRS sub-scores aren't in there

The raw data hands you `Hygiene`, `Structural`, and `ConfidenceInManagement` — three ready-made numeric columns just sitting there. Didn't touch them.

Reason: these three literally get combined to produce the final rating. Using them as features would be like handing the model the answer along with the question — and in practice none of them would even exist yet at the point you'd want to make a prediction, before the inspection has happened.

## 4. Cutting down the time window

Started with everything, going back to 1998. My supervisor pointed out old records probably don't say much about current conditions, so I cut it to the 5 years before extraction. Just `RatingDate >= max(RatingDate) - 5 years`.

## 5. Duplicates

Same business name + same postcode showing up twice = same place listed twice. Keep whichever has the more recent rating date, drop the other.

## 6. Deprivation data

Tried adding a local-authority-level deprivation score (from the IMD25 dataset), matched to each business by local authority name — had to hand-fix about 30 naming mismatches along the way (e.g. "Kingston upon Hull, City of" vs "Hull City" in the two datasets).

| Version | What it was | XGBoost AUC |
|---|---|---|
| v1 | No deprivation data | 0.774 |
| v2 (kept this one) | IMD25 score at local authority level, matched by name | 0.787 |

Went with v2 — a solid, consistent improvement over having no deprivation feature at all.


## 7. Dealing with the class imbalance

Used SMOTE, only on the training data — bumps the high-risk class up to a 50/50 split. Left the test set exactly as it was so the numbers I report still reflect the real-world imbalance.

Thought about just weighting the classes instead, but stuck with SMOTE to match what Allen et al. (2019) and Oldroyd et al. (2021) did, since those are my two closest comparison points.

## 8. One change that was just practical, not methodological

First try at Random Forest was 300 trees, no depth limit — this basically never finished running on the ~550K rows I had after SMOTE. Had to kill it.

Capped it at `max_depth=20`, 150 trees instead — runs in a few minutes, barely moved the results. Ran XGBoost with `tree_method='hist'` for the same reason, it's just faster at this size.

Not a methodology decision, just "this needed to actually finish running" — mentioning it because the settings in `fhrs_pipeline.py` are the fixed version, not what I started with.

## 9. How I picked the classification threshold

Used Youden's J (the ROC point that best balances catching true positives against false alarms) on top of the usual 0.5 cutoff. Picked this specifically because it's what Allen et al. (2019) and Oldroyd et al. (2021) both used, so my numbers actually compare to theirs properly.

## 10. Couple of mistakes I caught late and fixed

While going back through the dissertation against what the code actually spits out, found two things that didn't match:

- Said "298 local authorities" in one place and "297" everywhere else — fixed to 297 throughout, since that's what's actually in the data (see #2).
- The sample size (402,815) and test-set size (100,704) I'd written down were from *before* the deprivation merge — 8 records got dropped at that step for not matching a local authority. Fixed to the real numbers: 402,807 and 100,702.

Neither of these changed anything about how the data was actually coded — just fixed the write-up to match what the code does. Noting them here so the record's complete.

---

## If you want to check any of this

Every AUC number in section 6 came from actually running that version against the exact data sitting in `data/` (README has the details on when that was pulled). `fhrs_pipeline.py` here is the v2 setup — I didn't keep separate scripts for v1 since only v2 ended up in the dissertation.

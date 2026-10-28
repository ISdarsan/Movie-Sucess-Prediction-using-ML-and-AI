# Pre-release movie success prediction — comprehensive model comparison

*Generated from `backend/artifacts/comprehensive_results.json` (2026-10-26T17:48:36+00:00). Every number comes from that file. Seed 42; python 3.13.1, sklearn 1.9.1, pandas 3.0.6, numpy 2.5.4, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, imblearn 0.14.2.*

## 1. Objective

Predict whether a film will be a **Hit, Average or Flop** using only information available **before release**. The label is derived from the eventual IMDb score (`Flop` < 3 ≤ `Average` < 6 ≤ `Hit`, unchanged). It is a rating band, not box office or profit, and a prediction is an estimate, not a guarantee. This study re-opens the model choice: **no algorithm was assumed to win**, including the previously deployed LightGBM and the original Random Forest.

## 2. Verified setup (re-used unchanged)

| check | result |
|---|---|
| schema inputs equal feature set | True |
| excluded columns not inputs | True |
| target not input | True |
| split matches previous study | True |
| stratified | True |
| n inputs | 9 |

- Cleaned dataset: 5,043 raw rows → 4,998 after exact duplicates → **4,919** after removing the same film scraped twice (same title and year; films with a missing year are kept).
- Split: train_test_split(test_size=0.2, stratify=Classify, random_state=42) after cleaning (unchanged). Train 3,935 / test 984; the test set is untouched until the selection is frozen.

| class | all | train | test |
|---|---|---|---|
| Flop | 41 (0.8%) | 33 | 8 |
| Average | 1,383 (28.1%) | 1,106 | 277 |
| Hit | 3,495 (71.1%) | 2,796 | 699 |

**Class counts in every fold:** each training fold holds 26–27 Flops and each validation fold only **6–7** (of 33 in the whole training set). Each validation fold holds only a handful of Flops, so fold-level and test-level Flop metrics cannot be reliable; they are reported but not trusted.

## 3. The nine pre-release inputs and what was excluded

Inputs: `budget`, `duration`, `title_year`, `aspect_ratio`, `country`, `language`, `content_rating`, `color`, `genres`. Every other column was audited and excluded (gross, votes, reviews, social-media likes, plot keywords, identifiers, and the IMDb score itself):

| excluded column | timing | reason |
|---|---|---|
| `gross` | after release | Box-office revenue exists only after release. |
| `num_voted_users` | after release | IMDb votes accumulate after release (and track the target: Spearman 0.39 with IMDb score). |
| `num_user_for_reviews` | after release | User reviews are written after release. |
| `num_critic_for_reviews` | after release | Critic reviews are written after release. |
| `movie_facebook_likes` | after release | Social engagement with the film; collected in 2016, long after release. |
| `director_facebook_likes` | maybe before release | Popularity snapshot collected once (2016); its timing relative to each film cannot be established, and it can reflect that film's success. |
| `actor_1_facebook_likes` | maybe before release | Same as above: a 2016 snapshot, not a pre-release value. |
| `actor_2_facebook_likes` | maybe before release | Same as above. |
| `actor_3_facebook_likes` | maybe before release | Same as above. |
| `cast_total_facebook_likes` | maybe before release | Same as above; aggregate of the cast's 2016 popularity. |
| `facenumber_in_poster` | maybe before release | Poster artwork is a marketing asset whose timing is not recorded; weak signal, so excluded rather than assumed safe. |
| `plot_keywords` | after release | Keywords are user-generated on IMDb after release. |
| `movie_title` | identifier | Identifier. Used only to find near-duplicate rows. |
| `movie_imdb_link` | identifier | Identifier. |
| `imdb_score` | target | Used ONLY to create the Hit/Average/Flop label. Never an input. |

## 4. Preprocessing and imbalance strategies

All preprocessing lives inside the model pipeline and is fitted on training folds only: median imputation with missing flags, scaling and a log transform for budget, missing categoricals as their own `Unknown` level, rare categories pooled and unseen ones handled safely, genres as separate flags. Scaling is therefore always applied, which SVM, k-NN and logistic regression need.

| strategy | meaning |
|---|---|
| unweighted | the classifier sees the real class proportions |
| class weights | `class_weight='balanced'`, or balanced sample weights (Gradient Boosting, AdaBoost, XGBoost), CatBoost `auto_class_weights`, uniform priors for LDA |
| oversampling (in fold) | random oversampling of the minority classes, a step **inside** the CV pipeline so it only ever touches training folds |
| SMOTE (in fold) | synthetic interpolation after encoding, also inside the pipeline. Caveat: the encoded space mixes scaled numbers with 0/1 flags, so interpolated rows contain fractional flags — it is evaluated, not assumed suitable |

## 5. Algorithms evaluated

| algorithm | library | complexity rank | imbalance strategies run |
|---|---|---|---|
| Dummy baseline | scikit-learn | 0 | majority class, random, class proportions, random, uniform |
| Logistic Regression | scikit-learn | 1 | unweighted, class weights, oversampling (in fold), SMOTE (in fold) |
| Linear Discriminant Analysis | scikit-learn | 1 | unweighted, class weights |
| Decision Tree | scikit-learn | 2 | unweighted, class weights, oversampling (in fold) |
| k-Nearest Neighbours | scikit-learn | 2 | unweighted, oversampling (in fold) |
| Support Vector Machine (RBF) | scikit-learn | 3 | unweighted, class weights, oversampling (in fold) |
| Random Forest | scikit-learn | 4 | unweighted, class weights, oversampling (in fold), SMOTE (in fold) |
| Extra Trees | scikit-learn | 4 | unweighted, class weights, oversampling (in fold) |
| AdaBoost | scikit-learn | 4 | unweighted, class weights |
| Gradient Boosting | scikit-learn | 5 | unweighted, class weights, oversampling (in fold) |
| HistGradientBoosting | scikit-learn | 5 | unweighted, class weights, oversampling (in fold), SMOTE (in fold) |
| LightGBM | lightgbm | 5 | unweighted, class weights, oversampling (in fold), SMOTE (in fold) |
| XGBoost | xgboost | 5 | unweighted, class weights, oversampling (in fold) |
| CatBoost | catboost | 5 | unweighted, class weights |

**42 configurations** across 14 algorithms were executed.

Not evaluated as classifiers, and why:
- Linear Regression: A regressor, not a classifier. Evaluated only as a separate experiment (regression on the IMDb score, converted to classes) and not eligible for selection.

## 6. Protocol and selection rule (fixed before any test number was computed)

```
Stage 1  Screening.   Every algorithm x imbalance-strategy configuration, 5-fold x 3-repeat stratified CV on the TRAINING set (seed 42).
  Stage 2  Tuning.      The best configuration of each of the 5 strongest ALGORITHMS is tuned (randomised search, 5-fold, macro F1).
  Stage 3  Finalists.   The top-8 untuned configurations, the 5 tuned ones and the currently deployed LightGBM are re-scored on FRESH
                        folds (seed 44), identical for all, so they are compared on exactly the same splits and winner's-curse optimism
                        from stages 1-2 is not carried into the comparison.
  RULE     best = highest mean macro F1 on the stage-3 folds.  A finalist is "not distinguishable from best" if the Nadeau-Bengio
           corrected paired t-statistic of (best - finalist) is below 2.0.  Among the indistinguishable set choose the SIMPLEST model
           (fixed complexity ranking below); ties -> lower fold-to-fold std of macro F1 (more stable); then higher balanced accuracy.
           The test set, the temporal evaluation and the regression experiment play NO part in selection.
  Report   After the choice is frozen: held-out test metrics for every finalist and baseline, temporal evaluation, confusion matrices.
```

## 7. Stage 1 — screening of every configuration (cross-validation, training set only)

RepeatedStratifiedKFold(5 x 3, seed 42) on the training set. Sorted by mean macro F1.

| configuration | macro F1 (±sd) | balanced acc. | accuracy | Flop recall | Flop F1 | fit s |
|---|---|---|---|---|---|---|
| XGBoost · class weights | 0.488 ± 0.025 | 0.520 | 71.9% | 13% | 0.08 | 1.0 |
| Random Forest · oversampling (in fold) | 0.487 ± 0.038 | 0.488 | 75.1% | 4% | 0.05 | 3.1 |
| XGBoost · oversampling (in fold) | 0.485 ± 0.020 | 0.519 | 71.4% | 15% | 0.09 | 1.8 |
| HistGradientBoosting · oversampling (in fold) | 0.485 ± 0.033 | 0.492 | 72.8% | 6% | 0.07 | 3.4 |
| CatBoost · class weights | 0.483 ± 0.026 | 0.504 | 71.9% | 8% | 0.06 | 5.3 |
| LightGBM · oversampling (in fold) | 0.483 ± 0.031 | 0.496 | 72.4% | 7% | 0.07 | 0.9 |
| Random Forest · class weights | 0.481 ± 0.031 | 0.495 | 72.8% | 5% | 0.05 | 1.7 |
| Gradient Boosting · class weights | 0.480 ± 0.021 | 0.519 | 70.9% | 15% | 0.07 | 3.5 |
| LightGBM · SMOTE (in fold) | 0.480 ± 0.040 | 0.476 | 74.9% | 5% | 0.06 | 2.0 |
| HistGradientBoosting · SMOTE (in fold) | 0.480 ± 0.039 | 0.475 | 74.9% | 5% | 0.06 | 4.3 |
| LightGBM · class weights | 0.479 ± 0.031 | 0.493 | 72.1% | 6% | 0.06 | 0.7 |
| Gradient Boosting · oversampling (in fold) | 0.478 ± 0.022 | 0.524 | 70.3% | 18% | 0.07 | 7.9 |
| Random Forest · SMOTE (in fold) | 0.476 ± 0.033 | 0.476 | 75.1% | 3% | 0.03 | 3.7 |
| HistGradientBoosting · class weights | 0.471 ± 0.020 | 0.484 | 72.0% | 4% | 0.04 | 2.5 |
| Support Vector Machine (RBF) · oversampling (in fold) | 0.469 ± 0.021 | 0.522 | 69.1% | 20% | 0.07 | 22.7 |
| Support Vector Machine (RBF) · class weights | 0.467 ± 0.019 | 0.526 | 68.4% | 22% | 0.07 | 5.7 |
| Extra Trees · oversampling (in fold) | 0.466 ± 0.025 | 0.472 | 72.4% | 5% | 0.04 | 3.7 |
| Extra Trees · class weights | 0.458 ± 0.024 | 0.475 | 70.3% | 6% | 0.03 | 2.2 |
| XGBoost · unweighted | 0.457 ± 0.013 | 0.451 | 76.4% | 0% | 0.00 | 1.8 |
| HistGradientBoosting · unweighted | 0.457 ± 0.018 | 0.450 | 75.5% | 1% | 0.01 | 2.3 |
| LightGBM · unweighted | 0.456 ± 0.008 | 0.450 | 76.1% | 0% | 0.00 | 0.7 |
| CatBoost · unweighted | 0.456 ± 0.010 | 0.448 | 76.6% | 0% | 0.00 | 5.8 |
| Gradient Boosting · unweighted | 0.453 ± 0.013 | 0.446 | 76.1% | 0% | 0.00 | 3.5 |
| Linear Discriminant Analysis · class weights | 0.450 ± 0.016 | 0.488 | 66.6% | 11% | 0.04 | 0.1 |
| Random Forest · unweighted | 0.447 ± 0.011 | 0.439 | 76.5% | 0% | 0.00 | 1.8 |
| Decision Tree · unweighted | 0.446 ± 0.012 | 0.446 | 73.4% | 0% | 0.00 | 0.1 |
| k-Nearest Neighbours · unweighted | 0.444 ± 0.012 | 0.438 | 75.1% | 0% | 0.00 | 0.1 |
| AdaBoost · class weights | 0.441 ± 0.026 | 0.549 | 64.2% | 42% | 0.07 | 2.0 |
| Support Vector Machine (RBF) · unweighted | 0.439 ± 0.011 | 0.432 | 75.8% | 0% | 0.00 | 4.5 |
| Logistic Regression · SMOTE (in fold) | 0.438 ± 0.011 | 0.522 | 64.4% | 33% | 0.05 | 0.7 |
| Logistic Regression · unweighted | 0.437 ± 0.012 | 0.430 | 75.0% | 0% | 0.00 | 0.3 |
| k-Nearest Neighbours · oversampling (in fold) | 0.436 ± 0.013 | 0.468 | 65.1% | 8% | 0.02 | 0.1 |
| Logistic Regression · oversampling (in fold) | 0.433 ± 0.015 | 0.510 | 63.6% | 30% | 0.04 | 0.7 |
| AdaBoost · unweighted | 0.433 ± 0.012 | 0.428 | 74.4% | 0% | 0.00 | 2.0 |
| Extra Trees · unweighted | 0.432 ± 0.010 | 0.425 | 75.5% | 0% | 0.00 | 1.9 |
| Logistic Regression · class weights | 0.432 ± 0.014 | 0.502 | 63.5% | 28% | 0.04 | 0.4 |
| Linear Discriminant Analysis · unweighted | 0.426 ± 0.021 | 0.419 | 73.9% | 1% | 0.01 | 0.1 |
| Decision Tree · oversampling (in fold) | 0.409 ± 0.019 | 0.536 | 59.3% | 49% | 0.06 | 0.1 |
| Decision Tree · class weights | 0.403 ± 0.014 | 0.534 | 58.0% | 50% | 0.05 | 0.1 |
| Dummy baseline · random, class proportions | 0.329 ± 0.009 | 0.329 | 57.7% | 0% | 0.00 | 0.1 |
| Dummy baseline · majority class | 0.277 ± 0.000 | 0.333 | 71.1% | 0% | 0.00 | 0.1 |
| Dummy baseline · random, uniform | 0.257 ± 0.011 | 0.309 | 33.1% | 24% | 0.01 | 0.1 |

**Effect of the imbalance strategy** (mean over every algorithm that used it; cross-validation, training set):

| strategy | configs | mean macro F1 | mean balanced acc. | mean accuracy | mean Flop recall |
|---|---|---|---|---|---|
| unweighted | 13 | 0.445 | 0.439 | 75.4% | 0% |
| class weights | 12 | 0.461 | 0.508 | 68.5% | 18% |
| oversampling (in fold) | 10 | 0.463 | 0.503 | 69.1% | 16% |
| SMOTE (in fold) | 4 | 0.468 | 0.487 | 72.3% | 11% |

Balancing raises macro F1, balanced accuracy and Flop recall and lowers plain accuracy — a genuine trade-off, not a free improvement.

## 8. Stage 2 — tuning the five strongest algorithms

| tuned configuration | search CV macro F1 | best parameters |
|---|---|---|
| XGBoost · class weights · tuned | 0.480 (untuned 0.488) | subsample=1.0, reg_lambda=5, n_estimators=400, min_child_weight=3, max_depth=4, learning_rate=0.1, colsample_bytree=0.6 |
| Random Forest · oversampling (in fold) · tuned | 0.495 (untuned 0.487) | n_estimators=300, min_samples_leaf=3, max_features=sqrt, max_depth=None |
| HistGradientBoosting · oversampling (in fold) · tuned | 0.488 (untuned 0.485) | min_samples_leaf=80, max_leaf_nodes=31, max_iter=100, learning_rate=0.04, l2_regularization=10.0 |
| CatBoost · class weights · tuned | 0.488 (untuned 0.483) | learning_rate=0.06, l2_leaf_reg=1, iterations=200, depth=6 |
| LightGBM · oversampling (in fold) · tuned | 0.485 (untuned 0.483) | reg_lambda=1.0, num_leaves=15, n_estimators=250, min_child_samples=40, learning_rate=0.04, colsample_bytree=0.6 |

## 9. Stage 3 — finalists on identical fresh folds, and the selection

RepeatedStratifiedKFold(5 x 3, seed 44) — fresh folds, identical for every finalist. Winner's-curse optimism from stages 1–2 is not carried over.

| finalist | macro F1 (±sd) | balanced acc. | complexity | paired t vs best | |
|---|---|---|---|---|---|
| XGBoost · oversampling (in fold) | 0.493 ± 0.030 | 0.532 | 5 | 0.0 | best by mean |
| Random Forest · oversampling (in fold) · tuned | 0.493 ± 0.037 | 0.498 | 4 | 0.005 |  |
| LightGBM · oversampling (in fold) · tuned | 0.492 ± 0.036 | 0.507 | 5 | 0.041 |  |
| XGBoost · class weights | 0.492 ± 0.029 | 0.526 | 5 | 0.05 |  |
| LightGBM · class weights | 0.491 ± 0.034 | 0.505 | 5 | 0.104 | deployed before |
| Random Forest · oversampling (in fold) | 0.490 ± 0.037 | 0.491 | 4 | 0.115 |  |
| Random Forest · class weights | 0.489 ± 0.029 | 0.503 | 4 | 0.28 | **← selected** |
| LightGBM · oversampling (in fold) | 0.489 ± 0.034 | 0.499 | 5 | 0.275 |  |
| HistGradientBoosting · oversampling (in fold) · tuned | 0.488 ± 0.031 | 0.508 | 5 | 0.292 |  |
| XGBoost · class weights · tuned | 0.486 ± 0.029 | 0.505 | 5 | 0.396 |  |
| CatBoost · class weights | 0.485 ± 0.034 | 0.509 | 5 | 0.606 |  |
| HistGradientBoosting · oversampling (in fold) | 0.484 ± 0.040 | 0.492 | 5 | 0.4 |  |
| Gradient Boosting · class weights | 0.484 ± 0.025 | 0.525 | 5 | 0.547 |  |
| CatBoost · class weights · tuned | 0.482 ± 0.033 | 0.512 | 5 | 1.159 |  |

- Best by mean macro F1: **XGBoost · oversampling (in fold)**.
- Not distinguishable from the best (corrected paired t < 2): `XGBoost · class weights`, `Random Forest · oversampling (in fold)`, `XGBoost · oversampling (in fold)`, `HistGradientBoosting · oversampling (in fold)`, `CatBoost · class weights`, `LightGBM · oversampling (in fold)`, `Random Forest · class weights`, `Gradient Boosting · class weights`, `XGBoost · class weights · tuned`, `Random Forest · oversampling (in fold) · tuned`, `HistGradientBoosting · oversampling (in fold) · tuned`, `CatBoost · class weights · tuned`, `LightGBM · oversampling (in fold) · tuned`, `LightGBM · class weights`.
- **Chosen by the rule (simplest among the indistinguishable): Random Forest · class weights** (complexity rank 4). Outcome relative to the previously deployed `LightGBM · class weights`: **differs from deployed**.

## 10. Held-out test results (computed once, after the selection was frozen)

| algorithm | imbalance strategy | CV macro F1 | CV sd | test acc. | test bal. acc. | test macro F1 | Flop P / R / F1 | Average P / R / F1 | Hit P / R / F1 | fit s | predict ms/1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | oversampling (in fold) | 0.493 | 0.030 | 70.8% | 55.4% | 0.501 | 0.09 / 0.25 / 0.13 | 0.51 / 0.69 / 0.58 | 0.86 / 0.72 / 0.79 | 0.91 | 14.25 |
| Random Forest (tuned) | oversampling (in fold) | 0.493 | 0.037 | 75.0% | 53.0% | 0.543 | 0.50 / 0.12 / 0.20 | 0.55 / 0.68 / 0.61 | 0.86 / 0.79 / 0.82 | 1.44 | 43.21 |
| LightGBM (tuned) | oversampling (in fold) | 0.492 | 0.036 | 72.0% | 51.7% | 0.501 | 0.12 / 0.12 / 0.12 | 0.51 / 0.69 / 0.58 | 0.86 / 0.74 / 0.79 | 0.53 | 57.24 |
| XGBoost | class weights | 0.492 | 0.029 | 71.1% | 51.5% | 0.483 | 0.05 / 0.12 / 0.07 | 0.51 / 0.70 / 0.59 | 0.87 / 0.72 / 0.79 | 0.53 | 16.02 |
| LightGBM *(deployed before)* | class weights | 0.491 | 0.034 | 71.5% | 51.2% | 0.491 | 0.09 / 0.12 / 0.11 | 0.50 / 0.67 / 0.57 | 0.86 / 0.74 / 0.79 | 0.29 | 51.5 |
| Random Forest | oversampling (in fold) | 0.490 | 0.037 | 75.6% | 52.3% | 0.549 | 1.00 / 0.12 / 0.22 | 0.56 / 0.63 / 0.60 | 0.84 / 0.81 / 0.83 | 1.46 | 48.32 |
| Random Forest **← selected** | class weights | 0.489 | 0.029 | 74.0% | 53.1% | 0.516 | 0.12 / 0.12 / 0.12 | 0.54 / 0.71 / 0.61 | 0.87 / 0.76 / 0.81 | 0.9 | 55.28 |
| LightGBM | oversampling (in fold) | 0.489 | 0.034 | 72.8% | 56.5% | 0.533 | 0.17 / 0.25 / 0.20 | 0.52 / 0.70 / 0.60 | 0.87 / 0.74 / 0.80 | 0.42 | 54.24 |
| HistGradientBoosting (tuned) | oversampling (in fold) | 0.488 | 0.031 | 72.3% | 51.5% | 0.491 | 0.07 / 0.12 / 0.09 | 0.52 / 0.67 / 0.58 | 0.86 / 0.75 / 0.80 | 0.98 | 19.65 |
| XGBoost (tuned) | class weights | 0.486 | 0.029 | 71.2% | 51.1% | 0.488 | 0.08 / 0.12 / 0.10 | 0.50 / 0.67 / 0.57 | 0.85 / 0.74 / 0.79 | 0.53 | 18.09 |
| CatBoost | class weights | 0.485 | 0.034 | 72.9% | 56.7% | 0.520 | 0.11 / 0.25 / 0.15 | 0.53 / 0.71 / 0.61 | 0.88 / 0.74 / 0.80 | 2.13 | 9.76 |
| HistGradientBoosting | oversampling (in fold) | 0.484 | 0.040 | 73.4% | 56.2% | 0.550 | 0.25 / 0.25 / 0.25 | 0.53 / 0.68 / 0.59 | 0.86 / 0.76 / 0.81 | 1.02 | 21.71 |
| Gradient Boosting | class weights | 0.484 | 0.025 | 70.2% | 51.1% | 0.473 | 0.03 / 0.12 / 0.05 | 0.50 / 0.70 / 0.58 | 0.87 / 0.71 / 0.78 | 2.13 | 12.25 |
| CatBoost (tuned) | class weights | 0.482 | 0.033 | 73.1% | 57.3% | 0.518 | 0.09 / 0.25 / 0.13 | 0.54 / 0.73 / 0.62 | 0.88 / 0.74 / 0.80 | 1.39 | 9.77 |
| Dummy baseline | majority class | 0.277 | 0.000 | 71.0% | 33.3% | 0.277 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.71 / 1.00 / 0.83 | 0.03 | 7.67 |
| Dummy baseline | random, class proportions | 0.329 | 0.009 | 60.3% | 34.8% | 0.348 | 0.00 / 0.00 / 0.00 | 0.32 / 0.32 / 0.32 | 0.72 / 0.72 / 0.72 | 0.03 | 7.46 |
| Dummy baseline | random, uniform | 0.257 | 0.011 | 33.1% | 39.2% | 0.260 | 0.01 / 0.50 / 0.02 | 0.28 / 0.36 / 0.32 | 0.71 / 0.32 / 0.44 | 0.03 | 6.98 |

**CV versus test.** The finalist with the highest test macro F1 is `HistGradientBoosting · oversampling (in fold)` (0.550); the selected model scored 0.516. They disagree, which is exactly what a ranking within the noise looks like: with 8 test Flops and a cross-validation spread of about ±0.03 the test set cannot adjudicate between finalists, and the selection rule was not changed after seeing it.
**Versus the previously deployed model** (`LightGBM · class weights`): selected accuracy 74.0% vs 71.5%, balanced accuracy 53.1% vs 51.2%, macro F1 0.516 vs 0.491, Flops found 1 vs 1 of 8.
**Versus always answering the majority class:** accuracy 74.0% vs 71.0%; balanced accuracy 53.1% vs 33.3%; macro F1 0.516 vs 0.277.

### Confusion matrices (rows = true, columns = predicted; Flop, Average, Hit)

- **XGBoost · oversampling (in fold)**: `[[2, 3, 3], [9, 192, 76], [11, 185, 503]]`
- **Random Forest · oversampling (in fold) · tuned**: `[[1, 4, 3], [0, 188, 89], [1, 149, 549]]`
- **LightGBM · oversampling (in fold) · tuned**: `[[1, 4, 3], [5, 190, 82], [2, 180, 517]]`
- **XGBoost · class weights**: `[[1, 4, 3], [9, 193, 75], [10, 183, 506]]`
- **LightGBM · class weights**: `[[1, 4, 3], [8, 186, 83], [2, 180, 517]]`
- **Random Forest · oversampling (in fold)**: `[[1, 4, 3], [0, 175, 102], [0, 131, 568]]`
- **Random Forest · class weights** ← selected: `[[1, 4, 3], [4, 197, 76], [3, 166, 530]]`
- **LightGBM · oversampling (in fold)**: `[[2, 3, 3], [7, 195, 75], [3, 177, 519]]`
- **HistGradientBoosting · oversampling (in fold) · tuned**: `[[1, 4, 3], [8, 185, 84], [5, 169, 525]]`
- **XGBoost · class weights · tuned**: `[[1, 4, 3], [6, 186, 85], [5, 180, 514]]`
- **CatBoost · class weights**: `[[2, 3, 3], [9, 197, 71], [7, 174, 518]]`
- **HistGradientBoosting · oversampling (in fold)**: `[[2, 3, 3], [4, 187, 86], [2, 164, 533]]`
- **Gradient Boosting · class weights**: `[[1, 4, 3], [14, 193, 70], [15, 187, 497]]`
- **CatBoost · class weights · tuned**: `[[2, 3, 3], [8, 203, 66], [12, 173, 514]]`

### Classification report — Random Forest · class weights

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Flop | 0.12 | 0.12 | 0.12 | 8 |
| Average | 0.54 | 0.71 | 0.61 | 277 |
| Hit | 0.87 | 0.76 | 0.81 | 699 |

Pooled out-of-fold recall on the training set (more Flop evidence than the test set): Flop 9%, Average 67%, Hit 75%.

## 11. Did Flop detection improve meaningfully?

No. The selected model finds 1 of 8 test Flops (precision 0.12). The finalist that flags the most Flops, `XGBoost · oversampling (in fold)`, finds 2 of 8 but only 9% of the films it calls Flop are Flops, and its accuracy is 70.8%. With 33 training Flops and 6–7 per validation fold, differences of a film or two are noise. No configuration is claimed to detect Flops reliably.

## 12. Temporal generalisation (reporting only; not used for selection)

Preprocessing and models are refit on the earlier period only. Films with a missing year are excluded. Reporting only: not used for selection.

| period | train / test films | test classes (F / A / H) | always-majority accuracy | Random Forest · class weights accuracy | bal. acc. | macro F1 | Flops found |
|---|---|---|---|---|---|---|---|
| train ≤ 2009 → test ≥ 2010 | 3,367 / 1,446 | 10 / 487 / 949 | 65.6% | 68.0% | 44.8% | 0.443 | 0 of 10 |
| train ≤ 2004 → test 2005–2008 | 2,242 / 872 | 13 / 235 / 624 | 71.6% | 70.1% | 47.9% | 0.452 | 0 of 13 |
| train ≤ 2008 → test 2009–2012 | 3,114 / 916 | 7 / 299 / 610 | 66.6% | 68.6% | 49.4% | 0.469 | 1 of 7 |
| train ≤ 2012 → test 2013–2016 | 4,030 / 783 | 6 / 267 / 510 | 65.1% | 68.8% | 46.0% | 0.452 | 0 of 6 |

Flops in the later windows number in the single digits, so temporal Flop figures are not interpretable; the accuracy and balanced-accuracy comparisons with the majority baseline are the usable evidence. Reliability on future films is **not** claimed from the random split.

Main temporal split, all finalists:

| finalist | accuracy | balanced acc. | macro F1 |
|---|---|---|---|
| XGBoost · class weights | 65.4% | 47.3% | 0.447 |
| Random Forest · oversampling (in fold) | 69.1% | 44.3% | 0.441 |
| XGBoost · oversampling (in fold) | 65.5% | 47.1% | 0.447 |
| HistGradientBoosting · oversampling (in fold) | 67.5% | 47.9% | 0.469 |
| CatBoost · class weights | 65.8% | 46.9% | 0.448 |
| LightGBM · oversampling (in fold) | 66.7% | 47.4% | 0.467 |
| Random Forest · class weights ← selected | 68.0% | 44.8% | 0.443 |
| Gradient Boosting · class weights | 64.2% | 46.0% | 0.435 |
| XGBoost · class weights · tuned | 64.9% | 46.9% | 0.450 |
| Random Forest · oversampling (in fold) · tuned | 67.8% | 44.2% | 0.437 |
| HistGradientBoosting · oversampling (in fold) · tuned | 66.9% | 48.2% | 0.462 |
| CatBoost · class weights · tuned | 66.5% | 50.4% | 0.460 |
| LightGBM · oversampling (in fold) · tuned | 65.6% | 46.9% | 0.458 |
| LightGBM · class weights | 67.1% | 48.2% | 0.472 |

## 13. Separate experiment: regression on the IMDb score

Predict the continuous IMDb score from the same 9 inputs, then convert with the target's own bins (below 3 = Flop, 3 to below 6 = Average, 6 and above = Hit). Same CV folds as stage 1. A separate experiment: not eligible for selection.

| regressor | macro F1 | balanced acc. | accuracy |
|---|---|---|---|
| Ridge regression | 0.448 ± 0.011 | 0.446 | 74.5% |
| Random Forest regressor | 0.468 ± 0.009 | 0.468 | 75.9% |
| HistGradientBoosting regressor | 0.470 ± 0.011 | 0.469 | 76.4% |

## 14. Limitations

1. Only 41 Flops exist; no model is reliable on that class, and the test set has 8.
2. Pre-release features give a modest signal: accuracy is close to the majority baseline; the advantage is in balanced accuracy.
3. The main split is random; the temporal checks suggest the signal is weaker for later films, and `title_year` lets models learn era effects.
4. The label is an IMDb-rating band, not commercial success; `budget` mixes currencies; `content_rating` is assigned late in production.
5. Many finalists are statistically tied, so the choice rests on a pre-declared simplicity rule, not on a decisive win.
6. Class scores are uncalibrated estimates; SHAP and importance describe the model, not causes.

## 15. Reproduce

```
.venv/Scripts/python -m backend.ml.run_comprehensive      # the whole comparison
.venv/Scripts/python -m pytest backend/tests/test_candidate.py -q   # integration tests on the candidate BEFORE promotion
.venv/Scripts/python -m backend.ml.promote_candidate     # promote only if those pass
.venv/Scripts/python -m backend.ml.export_site_artifacts # metrics + SHAP for the website
.venv/Scripts/python -m backend.ml.write_report          # this document
.venv/Scripts/python -m backend.ml.append_notebook_section
```

Earlier artifacts are kept: `pre_release_movie_classifier.v1_lightgbm.joblib` (the model deployed before this comparison), `prerelease_results_v1.json`, `model_metrics_v1.json`, `explain_v1.json`; and the original post-release model `final_movie_classifier.joblib`.

# Movie Success Prediction Using ML and AI

An in-progress movie success prediction project using movie metadata and IMDb-score-based categories: **Flop**, **Average**, and **Hit**. The notebook explores the data, preprocesses features, and compares baseline, linear, ensemble, and boosting classifiers using classification metrics and confusion matrices.

## Dataset

The `dataset/movie_metadata.csv` file is included in this repository. It is the IMDb 5000 Movie Dataset published on [Kaggle by Yueming](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset). The Kaggle listing identifies the database license as [Open Data Commons Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Please review the source listing and license for attribution, share-alike, and any rights applying to the individual contents.

## Getting started

Install the notebook dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn jupyter scikit-learn imbalanced-learn xgboost catboost lightgbm
```

Open and run `PHASE1.ipynb` from the project root:

```bash
jupyter notebook PHASE1.ipynb
```

The notebook assigns labels from `imdb_score` using these ranges:

- **Flop:** scores from 0 up to (but not including) 3
- **Average:** scores from 3 up to (but not including) 6
- **Hit:** scores from 6 up to (but not including) 10

The IMDb score and other identifier/text fields are excluded from the initial feature set.

## Pre-release prediction study

The original notebook cells classify films with columns that only exist *after* release (votes, reviews, gross, social-media likes). The pre-release sections at the end of `PHASE1.ipynb` (code in `backend/ml/`) answer the capstone question properly: *predict Hit / Average / Flop using only information known before release* (nine inputs: budget, duration, release year, aspect ratio, genres, country, language, content rating, colour).

- **Comprehensive comparison:** 14 algorithms (42 algorithm × imbalance-strategy configurations) screened with repeated stratified cross-validation, the five strongest tuned, 14 finalists re-scored on identical fresh folds, under a selection rule fixed before any test number was computed. Full write-up (generated from the saved results): [docs/PRE_RELEASE_MODEL.md](docs/PRE_RELEASE_MODEL.md).
- **Result:** the finalists are **statistically indistinguishable** (14 of 14 within noise of the best), so the rule chose the simplest of them: **Random Forest · class weights**. This is a defensible trade-off, not a decisive win. (The first study had deployed `LightGBM · class weights`; it is kept as a rollback copy.)
- **Honest headline:** held-out accuracy 74.0% vs 71.0% for always answering "Hit"; balanced accuracy 53.1% (chance 33.3%); macro F1 0.516; 1 of 8 test Flops found. On later films (train ≤ 2009, test ≥ 2010) accuracy is 68.0% vs 65.6% for the majority class. Flop detection did **not** meaningfully improve, and with only 41 Flops no Flop figure is reliable.
- **Saved model:** `pre_release_movie_classifier.joblib` (one pipeline: preprocessing + classifier). Rollback: `pre_release_movie_classifier.v1_lightgbm.joblib`. The original post-release `final_movie_classifier.joblib` is kept untouched for comparison; the website does not use it.

```bash
.venv/Scripts/python -m backend.ml.run_comprehensive         # the whole comparison (~6 min); writes a CANDIDATE, never the live model
.venv/Scripts/python -m pytest backend/tests/test_candidate.py -q   # integration tests on the candidate
.venv/Scripts/python -m backend.ml.promote_candidate         # promotes it only if those tests pass (keeps a rollback copy)
.venv/Scripts/python -m backend.ml.export_site_artifacts     # metrics + SHAP for the website
.venv/Scripts/python -m backend.ml.write_report              # regenerate docs/PRE_RELEASE_MODEL.md
.venv/Scripts/python -m backend.ml.append_notebook_section   # rebuild the executed notebook sections; original cells untouched
```

> Several *original* notebook cells call `joblib.dump(..., "final_movie_classifier.joblib")`. Re-running the whole notebook replaces that old model file (the site is unaffected: it reads `pre_release_movie_classifier.joblib`).

## FRAME / Film Intelligence web app

A six-page interface over the study: Overview, Predictor, Model performance, Explainability, Explore movies, Methodology.

- `backend/app` — FastAPI service. Loads the pre-release pipeline once at start-up and **verifies its input columns against `backend/artifacts/prerelease_schema.json`** (it refuses to serve on a mismatch). Requests containing a post-release column are rejected.
- `backend/ml` — the study (feature audit, cleaning, transformers, experiments, reports).
- `frontend/` — React 19 + TypeScript + Vite + Tailwind 4 + Recharts + Motion.

### Run (two terminals, from the project root)

```bash
# 1. API  (http://127.0.0.1:8000)
.venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m uvicorn backend.app.main:app --port 8000

# 2. Web  (http://localhost:5173, proxies /api to the API)
cd frontend && npm install && npm run dev
```

Production check: `cd frontend && npm run build && npm run preview` (port 4173). Serving the build for real (compression, caching, SPA fallback, `/api` proxy) is covered in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md); `vite preview` does not compress, so Lighthouse scores against it are pessimistic.

### Tests

```bash
.venv/Scripts/python -m pytest backend/tests -q
```

Covers the API contract, the comprehensive-comparison integrity checks (every algorithm actually executed, selection re-derived from the recorded fold scores, selection code never reads test data) and the ML correctness checks: audit completeness, no post-release/target column reaching the model, cleaning, stratified disjoint split, train-only preprocessing, oversampling only at fit time, saved-model reproducibility, round-trip, and that the recorded selection follows the frozen rule without using the test set.

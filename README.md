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

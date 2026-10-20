# Movie Success Prediction Using ML and AI

An exploratory data analysis and machine-learning project that uses movie metadata to classify films into three IMDb-score-based success categories: **Flop**, **Average**, and **Hit**. The initial notebook examines data quality and distributions, removes duplicate rows, and prepares model features and labels.

## Dataset

Place `movie_metadata.csv` in the `dataset/` directory before running the notebook. The dataset is not included in this repository; obtain it from its original source and follow that source's terms.

## Getting started

Install the notebook dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn jupyter
```

Open and run `PHASE1.ipynb` from the project root:

```bash
jupyter notebook PHASE1.ipynb
```

The notebook assigns labels from `imdb_score` using these ranges:

- **Flop:** scores from 0 up to (but not including) 3
- **Average:** scores from 3 up to (but not including) 6
- **Hit:** scores from 6 up to 10

The IMDb score and other identifier/text fields are excluded from the initial feature set.

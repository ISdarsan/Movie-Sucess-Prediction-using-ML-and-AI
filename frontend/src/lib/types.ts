import type { MovieClass } from './format'

export interface Health {
  status: 'ok' | 'degraded'
  model: { loaded: boolean; estimator?: string; n_estimators?: number; classes?: string[]; n_input_features?: number; error_type?: string }
  dataset: { rows: number }
  artifacts: Record<string, boolean>
}

export interface FilterOptions {
  year_min: number; year_max: number
  genres: { value: string; count: number }[]
  countries: { value: string; count: number }[]
  content_ratings: { value: string; count: number }[]
}

export interface Overview {
  empty: boolean
  filters: Record<string, string | number>
  n_movies: number; total_movies: number
  kpis: {
    movies: number; mean_score: number; median_score: number
    class_counts: Record<MovieClass, number>; class_share: Record<MovieClass, number>
    median_budget_usd: number | null; median_gross_usd: number | null
    budget_known: number; gross_known: number; usa_movies: number
  }
  class_distribution: { class: MovieClass; count: number; share: number }[]
  year_trend: ({ year: number; count: number; mean_score: number } & Record<MovieClass, number>)[]
  genres: ({ genre: string; count: number; mean_score: number } & Record<MovieClass, number>)[]
  score_histogram: { lo: number; hi: number; count: number; class: MovieClass }[]
  countries: { country: string; count: number; mean_score: number }[]
  budget_gross: { points: { id: number; title: string; year: number | null; budget: number; gross: number; class: MovieClass; score: number }[]; n_plotted: number; n_eligible: number; note: string }
  correlations: { feature: string; rho: number; n: number; usa_only: boolean }[]
  insights: string[]
  source: { dataset: string; rows_after_dedup: number; year_span: [number, number] }
}

export interface ClassReport { precision: number; recall: number; 'f1-score': number; support: number }
export interface ModelResult {
  name: string; family: string; note: string; selected: boolean; matches_notebook: boolean | null
  algorithm?: string; strategy?: string; complexity?: number; deployed_before?: boolean; predict_ms_per_1000?: number
  tuned?: boolean; cv_macro_f1?: number; cv_macro_f1_std?: number; cv_balanced_accuracy?: number; cv_accuracy?: number; cv_flop_recall?: number; fit_seconds?: number
  accuracy: number; balanced_accuracy: number; macro_f1: number
  per_class: Record<MovieClass, ClassReport>
  confusion_matrix: number[][]
}
export interface ScreeningRow { name: string; algorithm: string; strategy: string; complexity: number; ok: boolean; macro_f1: number[]; balanced_accuracy: number[]; accuracy: number[]; flop_recall: number[]; flop_f1: number[]; fit_seconds: number }
export interface TemporalWindow { label: string; train_size: number; test_size: number; classes_test: Record<MovieClass, number>; majority_baseline: ModelResult; models: Record<string, ModelResult> }
export interface ModelMetrics {
  generated_at: string
  stage?: string
  screening?: ScreeningRow[]
  selection?: { rule: string; best_by_mean: string; indistinguishable_from_best: string[]; paired_vs_best: Record<string, { best_minus_this: number; corrected_se: number; t: number | null }>; chosen: string; chosen_complexity_rank: number; deployed_before: string; outcome: string }
  fold_audit?: { folds: number; flops_in_training_set: number; flops_per_training_fold: number[]; flops_per_validation_fold: number[]; conclusion: string }
  algorithms?: { key: string; label: string; library: string; complexity: number; strategies: string[]; evaluated: boolean; note: string }[]
  skipped?: { algorithm?: string; name?: string; reason?: string; error?: string }[]
  tuning?: { name: string; algorithm_label: string; base_config: string; search_cv_macro_f1: number; best_params: Record<string, unknown>; n_iter: number }[]
  regression_experiment?: { definition: string; rows: { name: string; macro_f1: number[]; balanced_accuracy: number[]; accuracy: number[] }[] }
  temporal?: { note: string; main: TemporalWindow; rolling: TemporalWindow[] }
  deployed_before?: { name: string }
  feature_audit?: { column: string; timing: string; used: string; reason: string }[]
  feature_set?: { use_people: boolean; inputs: string[]; paired_macro_f1_gain: number; rule: string; ablation: Record<string, { macro_f1: number[]; balanced_accuracy: number[]; accuracy: number[]; flop_recall: number[] }> }
  paired_vs_selected?: Record<string, { mean_diff_macro_f1: number; corrected_se: number; t: number | null; selected_better_in_folds: number }>
  pooled_oof_train?: ModelResult
  temporal_validation?: { train_through: number; test_from: number; train_size: number; test_size: number; classes_train: Record<MovieClass, number>; classes_test: Record<MovieClass, number>; model: ModelResult; majority_baseline: ModelResult; note: string }
  reference_post_release?: { description: string; cv_accuracy: number; cv_balanced_accuracy: number; cv_macro_f1: number; cv_flop_recall: number; test: ModelResult }
  evaluation: { split: string; train_size: number; test_size: number; train_class_counts: Record<MovieClass, number>; test_class_counts: Record<MovieClass, number>; labels: MovieClass[]; source: string; data: { raw_rows: number; after_exact_duplicate_rows: number; after_title_year_duplicates: number; classes_all: Record<MovieClass, number> }; cv_protocol?: string; selection_rule?: string; n_configurations?: number; n_algorithms?: number }
  selected_model: {
    name: string; params: Record<string, string | number | boolean | null>; n_features_in: number; n_transformed_features: number; algorithm?: string; strategy?: string
    tuning: { method: string; grid: Record<string, (string | number | null)[]>; best_cv_macro_f1: number; origin: string }
  }
  held_out: ModelResult
  comparison: ModelResult[]
  cross_validation: {
    origin: string
    five_fold_train: { protocol: string; rows: { name: string; accuracy: number; balanced_accuracy: number; macro_f1: number; macro_f1_std: number }[] }
    repeated: { protocol: string; rows: { name: string; macro_f1: number; macro_f1_std: number; balanced_accuracy: number; balanced_accuracy_std: number }[] }
  }
}

export interface ShapBlock {
  bar: { feature: string; mean_abs_shap: number }[]
  beeswarm: { feature: string; points: [number, number][] }[]
  grouped_by_column: { column: string; mean_abs_shap: number }[]
}
export interface Explain {
  generated_at: string
  method: { explainer: string; sample_size: number; sample_from: string; n_transformed_features: number; classes: string[]; shap_version: string; values_shape: number[] }
  feature_importance: { kind: string; top: { feature: string; importance: number }[]; grouped_by_column: { column: string; importance: number }[] }
  shap: Record<MovieClass, ShapBlock>
}

export interface PredictSchema {
  stage: string
  columns: string[]; numeric: string[]
  numeric_ranges: Record<string, { min: number; max: number; p01: number; p99: number; median: number; missing_pct: number }>
  categorical: Record<string, string[]>
  genres: string[]
  defaults_when_blank: { numeric_median: Record<string, number>; categorical: string }
  classes: MovieClass[]
  excluded: { column: string; timing: string; reason: string }[]
}
export interface PredictResult {
  label: MovieClass
  probabilities: Record<MovieClass, number>
  inputs: Record<string, string | number>
  imputed: { field: string; value: number | string; rule: string }[]
  warnings: { field: string; message: string }[]
  caveats: string[]
  model: { name: string; source: string; stage?: string }
}

export interface MovieRow {
  id: number; title: string; year: number | null; genres: string | null; country: string | null; content_rating: string | null
  duration: number | null; imdb_score: number; Classify: MovieClass; budget: number | null; gross: number | null
  num_voted_users: number | null; director_name: string | null
}
export interface MovieList { total: number; page: number; page_size: number; pages: number; rows: MovieRow[] }
export interface MovieDetail extends MovieRow {
  language: string | null; color: string | null; aspect_ratio: number | null
  num_critic_for_reviews: number | null; num_user_for_reviews: number | null
  actor_1_name: string | null; actor_2_name: string | null; actor_3_name: string | null
  movie_facebook_likes: number | null; cast_total_facebook_likes: number | null; director_facebook_likes: number | null
  plot_keywords: string | null; movie_imdb_link: string | null
  genre_list: string[]; keywords: string[]; budget_currency_note: string | null
}

export interface TimeStats {
  avg: number;
  min: number;
  max: number;
}

export interface ModelPerformance {
  model: string;
  count: number;
  avg_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  avg_age_gap: number;
}

export interface Analytics {
  total_predictions: number;
  total_progressions: number;
  week_predictions: number;
  week_progressions: number;
  average_predicted_age: number;
  average_confidence: number;
  gender_distribution: { gender: string; count: number }[];
  emotion_distribution: { emotion: string; count: number }[];
  daily_counts: { date: string; count: number }[];
  detector_distribution: { detector: string; count: number }[];
  prediction_time_stats: TimeStats;
  progression_time_stats: TimeStats;
  confidence_distribution: { range: string; count: number }[];
  model_distribution: { model: string; count: number }[];
  model_performance: ModelPerformance[];
  age_transformation_stats: { avg_current: number; avg_target: number };
}

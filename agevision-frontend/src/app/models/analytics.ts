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
}

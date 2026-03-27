export interface Prediction {
  id: number;
  user: number;
  image: string;
  image_url: string;
  predicted_age: number;
  confidence: number;
  gender: string;
  emotion: string;
  face_count: number;
  processing_time_ms: number;
  created_at: string;
}

export interface FaceRegion {
  x_pct: number;
  y_pct: number;
  w_pct: number;
  h_pct: number;
}

export interface FacePrediction {
  face_id: number;
  predicted_age: number;
  confidence: number;
  gender: string;
  emotion: string;
  race: string;
  face_region: FaceRegion;
}

export interface PredictionResponse {
  message: string;
  prediction: Prediction;
  faces: FacePrediction[];
}

import { FacePrediction } from './prediction';

/** A successful per-image entry returned by the batch endpoint. */
export interface BatchSuccessItem {
  file_index: number;
  filename: string;
  face_count: number;
  image_url: string;
  image_path: string;
  faces: FacePrediction[];
  error?: undefined;
}

/** A failed per-image entry — only `error` is populated. */
export interface BatchErrorItem {
  file_index: number;
  filename: string;
  error: string;
  faces?: undefined;
  face_count?: undefined;
  image_url?: undefined;
  image_path?: undefined;
}

export type BatchResultItem = BatchSuccessItem | BatchErrorItem;

export interface BatchPredictionResponse {
  message: string;
  batch_id: string | null;
  total_images: number;
  total_faces: number;
  processing_time_ms: number;
  results: BatchResultItem[];
}

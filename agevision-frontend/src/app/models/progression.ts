export interface PipelineStep {
  label: string;
  icon: string;
  status: 'pending' | 'running' | 'done' | 'error';
  time_ms?: number;
}

export interface AgingInsight {
  label: string;
  value: number;   // 0-100
  color: string;
}

export interface Progression {
  id: number;
  original_image: string;
  progressed_image: string | null;
  original_image_url: string | null;
  progressed_image_url: string | null;
  current_age: number;
  target_age: number;
  model_used: string;
  processing_time_ms: number;
  gender: string;
  pipeline_steps: string;
  aging_insights: string;
  pipeline_steps_parsed: PipelineStep[];
  aging_insights_parsed: AgingInsight[];
  created_at: string;
}

export interface ProgressionResponse {
  message: string;
  progression: Progression;
  steps: PipelineStep[];
  insights: AgingInsight[];
}

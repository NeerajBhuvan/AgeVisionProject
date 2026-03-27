export interface UserSettings {
  id?: number;
  user?: number;
  theme: string;
  default_model: string;
  notifications_enabled: boolean;
  auto_detect_faces: boolean;
  save_to_history: boolean;
  high_accuracy_mode: boolean;
  show_confidence: boolean;
  language: string;
  timezone: string;
  updated_at?: string;
}

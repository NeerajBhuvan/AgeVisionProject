import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { ApiService } from './api.service';
import { NotificationService } from './notification.service';
import { TimezoneService } from './timezone.service';

/** App-wide user settings (the ones that actually drive behaviour). */
export interface AppSettings {
  theme: string;
  default_model: string;          // aging model: 'sam' | 'fast_aging' | 'diffusion'
  notifications_enabled: boolean;  // in-app notifications
  save_to_history: boolean;
  show_confidence: boolean;
  timezone: string;
}

const DEFAULTS: AppSettings = {
  theme: 'dark',
  default_model: 'sam',
  notifications_enabled: true,
  save_to_history: true,
  show_confidence: true,
  timezone: 'Asia/Kolkata',
};

const VALID_MODELS = ['sam', 'fast_aging', 'diffusion'];

/**
 * Single source of truth for user preferences across the app. Loaded once after
 * login (shared), exposed as an observable + sync getters, and persisted back
 * to the API. On every change it syncs the dependent services (notifications,
 * timezone) so the toggles actually do something.
 */
@Injectable({ providedIn: 'root' })
export class SettingsService {
  private subject = new BehaviorSubject<AppSettings>({ ...DEFAULTS });
  settings$ = this.subject.asObservable();
  private fetched = false;

  constructor(
    private api: ApiService,
    private notif: NotificationService,
    private tz: TimezoneService
  ) {}

  get current(): AppSettings { return this.subject.value; }
  get showConfidence(): boolean { return this.subject.value.show_confidence; }
  get saveToHistory(): boolean { return this.subject.value.save_to_history; }
  get defaultModel(): string { return this.subject.value.default_model; }

  /**
   * Trigger a one-time fetch into the live subject (idempotent), and return the
   * LIVE settings stream so callers always see the current value — including
   * changes made via save() after the initial load.
   */
  ensureLoaded(): Observable<AppSettings> {
    if (!this.fetched) {
      this.fetched = true;
      this.api.getSettings().subscribe({
        next: (data: any) => this.apply(data),
        error: () => { this.fetched = false; },
      });
    }
    return this.settings$;
  }

  /** Force a fresh load (e.g. right after login, possibly a different user). */
  reload(): void {
    this.fetched = true;
    this.api.getSettings().subscribe({
      next: (data: any) => this.apply(data),
      error: () => { this.fetched = false; },
    });
  }

  /** Persist a partial change; updates the app immediately (optimistic). */
  save(partial: Partial<AppSettings>): Observable<any> {
    const merged = { ...this.subject.value, ...partial };
    this.apply(merged);
    return this.api.updateSettings(merged);
  }

  private apply(data: any): void {
    const merged: AppSettings = { ...DEFAULTS, ...this.subject.value, ...(data || {}) };
    // Guard against legacy/fake model values that are no longer offered.
    if (!VALID_MODELS.includes(merged.default_model)) merged.default_model = 'sam';
    this.subject.next(merged);
    // Wire the toggles to real behaviour:
    this.notif.setEnabled(merged.notifications_enabled !== false);
    if (merged.timezone) this.tz.timezone = merged.timezone;
  }
}

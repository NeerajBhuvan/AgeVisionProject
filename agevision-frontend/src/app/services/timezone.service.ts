import { Injectable } from '@angular/core';
import { ApiService } from './api.service';

@Injectable({ providedIn: 'root' })
export class TimezoneService {
  private _timezone = 'Asia/Kolkata';

  /** All IANA timezones grouped by region for the dropdown. */
  static readonly TIMEZONE_OPTIONS: { label: string; value: string }[] = [
    // Asia
    { label: '(GMT+5:30) India - Kolkata', value: 'Asia/Kolkata' },
    { label: '(GMT+5:45) Nepal - Kathmandu', value: 'Asia/Kathmandu' },
    { label: '(GMT+6:00) Bangladesh - Dhaka', value: 'Asia/Dhaka' },
    { label: '(GMT+5:00) Pakistan - Karachi', value: 'Asia/Karachi' },
    { label: '(GMT+8:00) Singapore', value: 'Asia/Singapore' },
    { label: '(GMT+8:00) China - Shanghai', value: 'Asia/Shanghai' },
    { label: '(GMT+9:00) Japan - Tokyo', value: 'Asia/Tokyo' },
    { label: '(GMT+9:00) Korea - Seoul', value: 'Asia/Seoul' },
    { label: '(GMT+7:00) Thailand - Bangkok', value: 'Asia/Bangkok' },
    { label: '(GMT+3:00) Saudi Arabia - Riyadh', value: 'Asia/Riyadh' },
    { label: '(GMT+4:00) UAE - Dubai', value: 'Asia/Dubai' },
    { label: '(GMT+3:30) Iran - Tehran', value: 'Asia/Tehran' },
    // Americas
    { label: '(GMT-5:00) US Eastern - New York', value: 'America/New_York' },
    { label: '(GMT-6:00) US Central - Chicago', value: 'America/Chicago' },
    { label: '(GMT-7:00) US Mountain - Denver', value: 'America/Denver' },
    { label: '(GMT-8:00) US Pacific - Los Angeles', value: 'America/Los_Angeles' },
    { label: '(GMT-3:00) Brazil - Sao Paulo', value: 'America/Sao_Paulo' },
    { label: '(GMT-5:00) Canada - Toronto', value: 'America/Toronto' },
    { label: '(GMT-8:00) Canada - Vancouver', value: 'America/Vancouver' },
    // Europe
    { label: '(GMT+0:00) UK - London', value: 'Europe/London' },
    { label: '(GMT+1:00) France - Paris', value: 'Europe/Paris' },
    { label: '(GMT+1:00) Germany - Berlin', value: 'Europe/Berlin' },
    { label: '(GMT+3:00) Russia - Moscow', value: 'Europe/Moscow' },
    { label: '(GMT+2:00) Greece - Athens', value: 'Europe/Athens' },
    { label: '(GMT+1:00) Italy - Rome', value: 'Europe/Rome' },
    // Africa
    { label: '(GMT+1:00) Nigeria - Lagos', value: 'Africa/Lagos' },
    { label: '(GMT+2:00) South Africa - Johannesburg', value: 'Africa/Johannesburg' },
    { label: '(GMT+3:00) Kenya - Nairobi', value: 'Africa/Nairobi' },
    { label: '(GMT+2:00) Egypt - Cairo', value: 'Africa/Cairo' },
    // Oceania
    { label: '(GMT+11:00) Australia - Sydney', value: 'Australia/Sydney' },
    { label: '(GMT+8:00) Australia - Perth', value: 'Australia/Perth' },
    { label: '(GMT+13:00) New Zealand - Auckland', value: 'Pacific/Auckland' },
    // UTC
    { label: '(GMT+0:00) UTC', value: 'UTC' },
  ];

  constructor(private api: ApiService) {
    this.loadTimezone();
  }

  /** Current user timezone (IANA string). */
  get timezone(): string {
    return this._timezone;
  }

  /** Set timezone (called from settings). */
  set timezone(tz: string) {
    this._timezone = tz;
  }

  /** Load timezone from saved settings. */
  loadTimezone(): void {
    this.api.getSettings().subscribe({
      next: (data: any) => {
        if (data?.timezone) {
          this._timezone = data.timezone;
        }
      },
      error: () => {}
    });
  }

  /**
   * Format a Date or ISO string in the user's selected timezone.
   * Returns a human-readable relative string (e.g. "Just now", "5 min ago")
   * or a full formatted date for older items.
   */
  formatTimeAgo(date: Date | string): string {
    const d = date instanceof Date ? date : new Date(date);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);

    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin} min ago`;
    const diffHrs = Math.floor(diffMin / 60);
    if (diffHrs < 24) return `${diffHrs}h ago`;
    const diffDays = Math.floor(diffHrs / 24);
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;

    return this.formatDate(d);
  }

  /**
   * Format a Date or ISO string as a full date+time in the user's timezone.
   * Example: "Mar 9, 2026, 2:30 PM"
   */
  formatDateTime(date: Date | string): string {
    const d = date instanceof Date ? date : new Date(date);
    try {
      return d.toLocaleString('en-US', {
        timeZone: this._timezone,
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch {
      return d.toLocaleString();
    }
  }

  /**
   * Format a Date or ISO string as date only in the user's timezone.
   * Example: "Mar 9, 2026"
   */
  formatDate(date: Date | string): string {
    const d = date instanceof Date ? date : new Date(date);
    try {
      return d.toLocaleDateString('en-US', {
        timeZone: this._timezone,
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return d.toLocaleDateString();
    }
  }

  /**
   * Format a Date or ISO string as time only in the user's timezone.
   * Example: "2:30 PM"
   */
  formatTime(date: Date | string): string {
    const d = date instanceof Date ? date : new Date(date);
    try {
      return d.toLocaleTimeString('en-US', {
        timeZone: this._timezone,
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch {
      return d.toLocaleTimeString();
    }
  }

  /**
   * Get the day name (Mon, Tue, etc.) for a date string in user's timezone.
   */
  getDayLabel(dateStr: string): string {
    const d = new Date(dateStr);
    try {
      return d.toLocaleDateString('en-US', {
        timeZone: this._timezone,
        weekday: 'short',
      });
    } catch {
      return ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()];
    }
  }

  /**
   * Get current time formatted in the user's timezone.
   */
  getCurrentTime(): string {
    return this.formatTime(new Date());
  }

  /**
   * Get the timezone abbreviation (e.g. IST, EST, PST).
   */
  getTimezoneAbbr(): string {
    try {
      const parts = new Intl.DateTimeFormat('en-US', {
        timeZone: this._timezone,
        timeZoneName: 'short',
      }).formatToParts(new Date());
      const tzPart = parts.find(p => p.type === 'timeZoneName');
      return tzPart?.value || this._timezone;
    } catch {
      return this._timezone;
    }
  }
}

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ApiService } from '../../services/api.service';
import { ThemeService } from '../../services/theme.service';
import { TimezoneService } from '../../services/timezone.service';
import { UserSettings } from '../../models/user-settings';

@Component({
  selector: 'app-settings',
  imports: [CommonModule, FormsModule],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss'
})
export class SettingsComponent implements OnInit {
  activeSection = 'profile';

  sections = [
    { id: 'profile', label: 'Profile', icon: 'fa-solid fa-user' },
    { id: 'preferences', label: 'Preferences', icon: 'fa-solid fa-palette' },
    { id: 'security', label: 'Security', icon: 'fa-solid fa-lock' },
    { id: 'about', label: 'About', icon: 'fa-solid fa-circle-info' }
  ];

  // Profile
  profileName = '';
  profileEmail = '';
  profileSaving = false;
  profileMsg = '';

  // Preferences
  settings: UserSettings = {
    theme: 'dark',
    default_model: 'DeepFace v3',
    notifications_enabled: true,
    auto_detect_faces: true,
    save_to_history: true,
    high_accuracy_mode: false,
    show_confidence: true,
    language: 'English',
    timezone: 'Asia/Kolkata'
  };
  prefSaving = false;
  prefMsg = '';

  // Timezone options for dropdown
  timezoneOptions = TimezoneService.TIMEZONE_OPTIONS;

  // Security
  currentPassword = '';
  newPassword = '';
  confirmPassword = '';
  securitySaving = false;
  securityMsg = '';
  securityError = false;

  constructor(
    public auth: AuthService,
    private api: ApiService,
    private themeService: ThemeService,
    private timezoneService: TimezoneService,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    this.loadProfile();
    this.loadSettings();
    this.route.queryParams.subscribe(params => {
      if (params['section']) {
        setTimeout(() => this.setSection(params['section']), 100);
      }
    });
  }

  setSection(id: string): void {
    this.activeSection = id;
    const el = document.getElementById('settings-' + id);
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // -- Profile --
  private loadProfile(): void {
    const user = this.auth.getCurrentUser();
    if (user) {
      this.profileName = `${user.first_name || ''} ${user.last_name || ''}`.trim();
      this.profileEmail = user.email || '';
    }
  }

  saveProfile(): void {
    this.profileSaving = true;
    this.profileMsg = '';
    const parts = this.profileName.trim().split(/\s+/);
    const first_name = parts[0] || '';
    const last_name = parts.slice(1).join(' ') || '';

    this.api.updateProfile({ first_name, last_name, email: this.profileEmail }).subscribe({
      next: () => {
        this.profileMsg = 'Profile updated successfully';
        this.profileSaving = false;
        this.auth.loadUserProfile();
        setTimeout(() => this.profileMsg = '', 3000);
      },
      error: () => {
        this.profileMsg = 'Failed to update profile';
        this.profileSaving = false;
        setTimeout(() => this.profileMsg = '', 3000);
      }
    });
  }

  // -- Preferences --
  private loadSettings(): void {
    this.api.getSettings().subscribe({
      next: (data: UserSettings) => {
        this.settings = { ...this.settings, ...data };
        // Sync timezone to the service
        if (data.timezone) {
          this.timezoneService.timezone = data.timezone;
        }
      },
      error: () => {}
    });
  }

  togglePref(key: keyof UserSettings): void {
    (this.settings as any)[key] = !(this.settings as any)[key];
    this.savePreferences();

    if (key === 'theme' || key === 'notifications_enabled') {
      // Theme toggle is handled separately via the dedicated toggle
    }
  }

  onModelChange(): void {
    this.savePreferences();
  }

  onLanguageChange(): void {
    this.savePreferences();
  }

  onTimezoneChange(): void {
    this.timezoneService.timezone = this.settings.timezone;
    this.savePreferences();
  }

  private savePreferences(): void {
    this.prefSaving = true;
    this.prefMsg = '';
    const payload = {
      theme: this.settings.theme,
      default_model: this.settings.default_model,
      notifications_enabled: this.settings.notifications_enabled,
      auto_detect_faces: this.settings.auto_detect_faces,
      save_to_history: this.settings.save_to_history,
      high_accuracy_mode: this.settings.high_accuracy_mode,
      show_confidence: this.settings.show_confidence,
      language: this.settings.language,
      timezone: this.settings.timezone
    };

    this.api.updateSettings(payload).subscribe({
      next: () => {
        this.prefMsg = 'Preferences saved';
        this.prefSaving = false;
        setTimeout(() => this.prefMsg = '', 2000);
      },
      error: () => {
        this.prefMsg = 'Failed to save';
        this.prefSaving = false;
        setTimeout(() => this.prefMsg = '', 2000);
      }
    });
  }

  // -- Security --
  changePassword(): void {
    this.securityMsg = '';
    this.securityError = false;

    if (!this.currentPassword || !this.newPassword || !this.confirmPassword) {
      this.securityMsg = 'All fields are required';
      this.securityError = true;
      return;
    }
    if (this.newPassword.length < 8) {
      this.securityMsg = 'New password must be at least 8 characters';
      this.securityError = true;
      return;
    }
    if (this.newPassword !== this.confirmPassword) {
      this.securityMsg = 'Passwords do not match';
      this.securityError = true;
      return;
    }

    this.securitySaving = true;
    this.api.changePassword({
      current_password: this.currentPassword,
      new_password: this.newPassword
    }).subscribe({
      next: () => {
        this.securityMsg = 'Password changed successfully';
        this.securityError = false;
        this.securitySaving = false;
        this.currentPassword = '';
        this.newPassword = '';
        this.confirmPassword = '';
        setTimeout(() => this.securityMsg = '', 3000);
      },
      error: (err) => {
        this.securityMsg = err.error?.error || 'Failed to change password';
        this.securityError = true;
        this.securitySaving = false;
        setTimeout(() => this.securityMsg = '', 4000);
      }
    });
  }
}

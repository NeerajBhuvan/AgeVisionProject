import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export type Theme = 'dark' | 'light';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private readonly STORAGE_KEY = 'agevision-theme';

  /** BehaviorSubject holding current theme */
  private currentThemeSubject = new BehaviorSubject<Theme>(this.getStoredTheme());

  /** Observable for components to subscribe */
  currentTheme$ = this.currentThemeSubject.asObservable();

  constructor() {
    // Apply theme on service init
    this.applyTheme(this.currentThemeSubject.value);
  }

  /** Toggle between dark and light */
  toggleTheme(): void {
    const next: Theme = this.currentThemeSubject.value === 'dark' ? 'light' : 'dark';
    this.setTheme(next);
  }

  /** Set a specific theme */
  setTheme(theme: Theme): void {
    this.currentThemeSubject.next(theme);
    localStorage.setItem(this.STORAGE_KEY, theme);
    this.applyTheme(theme);
  }

  /** Get current theme value (synchronous) */
  getCurrentTheme(): Theme {
    return this.currentThemeSubject.value;
  }

  /** Check if current theme is dark */
  isDark(): boolean {
    return this.currentThemeSubject.value === 'dark';
  }

  /** Apply theme to the DOM */
  private applyTheme(theme: Theme): void {
    document.documentElement.setAttribute('data-theme', theme);
  }

  /** Read persisted theme or fall back to dark / OS preference */
  private getStoredTheme(): Theme {
    const stored = localStorage.getItem(this.STORAGE_KEY) as Theme | null;
    if (stored === 'dark' || stored === 'light') return stored;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
}

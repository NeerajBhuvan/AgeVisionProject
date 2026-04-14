import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../environments/environment';
import { NotificationService } from './notification.service';

/* ── Interfaces ── */

export interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  plan?: string;
  avatar_url?: string;
  is_active?: boolean;
  is_admin?: boolean;
  created_at?: string;
  updated_at?: string;
  last_login?: string;
}

/** Shape returned by the backend on login / register */
export interface AuthApiResponse {
  message: string;
  user: User;
  tokens: {
    access: string;
    refresh: string;
  };
}

/** Flattened shape used internally after mapping */
export interface AuthResponse {
  access: string;
  refresh: string;
  user: User;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  first_name: string;
  last_name: string;
}

export interface ForgotPasswordResponse {
  message: string;
  recovery_available?: boolean;
  password_hint?: string;
  password_length?: number;
  recovery_token?: string;
  username?: string;
}

export interface RecoverPasswordResponse {
  message: string;
  password?: string;
  username?: string;
}

/* ── Service ── */

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly baseUrl = environment.apiUrl;
  private readonly ACCESS_KEY = 'access_token';
  private readonly REFRESH_KEY = 'refresh_token';
  private readonly USER_KEY = 'agevision_user';

  /** BehaviorSubjects */
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  private isAuthenticatedSubject = new BehaviorSubject<boolean>(false);
  public isAuthenticated$ = this.isAuthenticatedSubject.asObservable();

  constructor(
    private http: HttpClient,
    private router: Router,
    private notifService: NotificationService
  ) {
    // Restore session if token exists
    const token = this.getAccessToken();
    if (token) {
      const storedUser = this.getStoredUser();
      if (storedUser) {
        this.currentUserSubject.next(storedUser);
        this.isAuthenticatedSubject.next(true);
        this.notifService.setUserScope(storedUser.id);
      }
      this.loadUserProfile();
    } else {
      this.notifService.setUserScope(null);
    }
  }

  /* ── Auth Actions ── */

  /** Login with username & password */
  login(credentials: LoginCredentials): Observable<AuthApiResponse> {
    return this.http
      .post<AuthApiResponse>(`${this.baseUrl}/auth/login/`, credentials)
      .pipe(
        tap(res => this.handleAuthSuccess(res))
      );
  }

  /** Register a new account */
  register(data: RegisterData): Observable<AuthApiResponse> {
    return this.http
      .post<AuthApiResponse>(`${this.baseUrl}/auth/register/`, data)
      .pipe(
        tap(res => this.handleAuthSuccess(res))
      );
  }

  /**
   * Refresh the access token using the stored refresh token.
   *
   * The backend has ROTATE_REFRESH_TOKENS=True, so the response carries a
   * brand-new refresh token alongside the new access token, and the old
   * refresh token is blacklisted server-side. We MUST persist the rotated
   * refresh token immediately — if we keep the old one, the next refresh
   * call will be rejected (blacklisted) and the user gets logged out.
   */
  refreshToken(): Observable<{ access: string; refresh?: string }> {
    const refresh = this.getRefreshToken();
    return this.http
      .post<{ access: string; refresh?: string }>(
        `${this.baseUrl}/auth/token/refresh/`,
        { refresh }
      )
      .pipe(
        tap(res => {
          localStorage.setItem(this.ACCESS_KEY, res.access);
          if (res.refresh) {
            localStorage.setItem(this.REFRESH_KEY, res.refresh);
          }
        })
      );
  }

  /** Logout — clear everything and navigate to login */
  logout(): void {
    this.clearTokens();
    localStorage.removeItem(this.USER_KEY);
    this.currentUserSubject.next(null);
    this.isAuthenticatedSubject.next(false);
    this.notifService.setUserScope(null);
    this.router.navigate(['/login']);
  }

  /* ── Password Recovery ── */

  /** Request password recovery by email */
  forgotPassword(email: string): Observable<ForgotPasswordResponse> {
    return this.http.post<ForgotPasswordResponse>(
      `${this.baseUrl}/auth/forgot-password/`, { email }
    );
  }

  /** Retrieve full decrypted password using recovery token */
  recoverPassword(email: string, recoveryToken: string): Observable<RecoverPasswordResponse> {
    return this.http.post<RecoverPasswordResponse>(
      `${this.baseUrl}/auth/recover-password/`,
      { email, recovery_token: recoveryToken }
    );
  }

  /** Reset password using recovery token */
  resetPassword(email: string, recoveryToken: string, newPassword: string): Observable<{ message: string }> {
    return this.http.post<{ message: string }>(
      `${this.baseUrl}/auth/reset-password/`,
      { email, recovery_token: recoveryToken, new_password: newPassword }
    );
  }

  /** Fetch profile from backend and update subjects */
  loadUserProfile(): void {
    // Let the auth interceptor attach the token — this way a refreshed
    // access token (after a 401) is picked up on the retry automatically.
    this.http
      .get<User>(`${this.baseUrl}/auth/profile/`)
      .subscribe({
        next: (user) => {
          this.currentUserSubject.next(user);
          this.isAuthenticatedSubject.next(true);
          this.notifService.setUserScope(user.id);
          localStorage.setItem(this.USER_KEY, JSON.stringify(user));
        },
        error: () => {
          // The interceptor already handles 401s (refresh + retry, or
          // full logout if refresh fails). Transient errors (network,
          // 5xx) must NOT clear the session — the cached user still
          // drives the UI until the next successful profile fetch.
        }
      });
  }

  /* ── Token helpers ── */

  getAccessToken(): string | null {
    return localStorage.getItem(this.ACCESS_KEY);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(this.REFRESH_KEY);
  }

  saveTokens(access: string, refresh: string): void {
    localStorage.setItem(this.ACCESS_KEY, access);
    localStorage.setItem(this.REFRESH_KEY, refresh);
  }

  clearTokens(): void {
    localStorage.removeItem(this.ACCESS_KEY);
    localStorage.removeItem(this.REFRESH_KEY);
  }

  /* ── Synchronous accessors ── */

  /** Check if user is authenticated (sync) */
  isAuthenticated(): boolean {
    return !!this.getAccessToken() && !!this.currentUserSubject.value;
  }

  /** Get current user snapshot */
  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  /** User initials for avatar (e.g. "NB") */
  getUserInitials(): string {
    const user = this.currentUserSubject.value;
    if (!user) return '?';
    const first = user.first_name?.[0] || user.username?.[0] || '';
    const last = user.last_name?.[0] || '';
    return (first + last).toUpperCase() || '?';
  }

  /** Display name (full name or username) */
  getDisplayName(): string {
    const user = this.currentUserSubject.value;
    if (!user) return 'Guest';
    if (user.first_name) {
      return `${user.first_name} ${user.last_name || ''}`.trim();
    }
    return user.username;
  }

  /* ── Private helpers ── */

  /** Handle successful login / register — map backend shape */
  private handleAuthSuccess(response: AuthApiResponse): void {
    this.saveTokens(response.tokens.access, response.tokens.refresh);
    this.currentUserSubject.next(response.user);
    this.isAuthenticatedSubject.next(true);
    localStorage.setItem(this.USER_KEY, JSON.stringify(response.user));
    this.notifService.setUserScope(response.user.id);
  }

  /** Build Authorization header */
  private authHeaders(): HttpHeaders {
    const token = this.getAccessToken();
    return new HttpHeaders({
      Authorization: `Bearer ${token}`
    });
  }

  /** Read locally-cached user */
  private getStoredUser(): User | null {
    try {
      const raw = localStorage.getItem(this.USER_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }
}

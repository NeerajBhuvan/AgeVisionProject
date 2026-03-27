import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('access_token');
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });
  }

  // Auth
  login(credentials: {username: string, password: string}): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/login/`, credentials);
  }

  register(data: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/register/`, data);
  }

  getProfile(): Observable<any> {
    return this.http.get(`${this.baseUrl}/auth/profile/`,
      { headers: this.getHeaders() });
  }

  // Core Features
  predictAge(formData: FormData): Observable<any> {
    return this.http.post(`${this.baseUrl}/predict/`, formData,
      { headers: this.getHeaders() });
  }

  predictCamera(frameBase64: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/predict/camera/`,
      { frame: frameBase64 },
      { headers: this.getHeaders().set('Content-Type', 'application/json') });
  }

  progressAge(formData: FormData): Observable<any> {
    return this.http.post(`${this.baseUrl}/progress/`, formData,
      { headers: this.getHeaders() });
  }

  // History
  getHistory(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/history/`,
      { headers: this.getHeaders() });
  }

  deleteHistory(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/history/${id}/`,
      { headers: this.getHeaders() });
  }

  // Analytics & Settings
  getAnalytics(): Observable<any> {
    return this.http.get(`${this.baseUrl}/analytics/`,
      { headers: this.getHeaders() });
  }

  getSettings(): Observable<any> {
    return this.http.get(`${this.baseUrl}/settings/`,
      { headers: this.getHeaders() });
  }

  updateSettings(settings: any): Observable<any> {
    return this.http.put(`${this.baseUrl}/settings/`, settings,
      { headers: this.getHeaders() });
  }

  updateProfile(data: any): Observable<any> {
    return this.http.put(`${this.baseUrl}/auth/profile/update/`, data,
      { headers: this.getHeaders() });
  }

  changePassword(data: { current_password: string; new_password: string }): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/change-password/`, data,
      { headers: this.getHeaders() });
  }
}

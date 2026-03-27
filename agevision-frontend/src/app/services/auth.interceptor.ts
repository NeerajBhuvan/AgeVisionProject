import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';
import { AuthService } from './auth.service';

/**
 * Functional HTTP interceptor that attaches the JWT access token
 * to every outgoing request and handles 401 responses.
 */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  // Skip auth header for login / register / token-refresh endpoints
  const openEndpoints = ['/auth/login/', '/auth/register/', '/auth/token/refresh/'];
  const isOpen = openEndpoints.some(ep => req.url.includes(ep));

  let authReq = req;
  if (!isOpen) {
    const token = authService.getAccessToken();
    if (token) {
      authReq = req.clone({
        setHeaders: { Authorization: `Bearer ${token}` }
      });
    }
  }

  return next(authReq).pipe(
    catchError((error: HttpErrorResponse) => {
      if (error.status === 401 && !isOpen) {
        // Token expired or invalid — force logout
        authService.logout();
      }
      return throwError(() => error);
    })
  );
};

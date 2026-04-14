import { HttpInterceptorFn, HttpErrorResponse, HttpRequest, HttpHandlerFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Observable, catchError, switchMap, throwError } from 'rxjs';
import { AuthService } from './auth.service';

/**
 * Functional HTTP interceptor that:
 *   1. Attaches the JWT access token to every outgoing request.
 *   2. On a 401 response, transparently uses the refresh token to obtain
 *      a new access token and replays the original request. Only if the
 *      refresh itself fails do we log the user out.
 *
 * This is what keeps the user signed in after a page reload once the
 * 24h access token has expired but the 7d refresh token is still valid.
 */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);

  // Skip auth header for login / register / token-refresh endpoints
  const openEndpoints = ['/auth/login/', '/auth/register/', '/auth/token/refresh/'];
  const isOpen = openEndpoints.some(ep => req.url.includes(ep));

  const authReq = isOpen ? req : attachToken(req, authService.getAccessToken());

  return next(authReq).pipe(
    catchError((error: HttpErrorResponse) => {
      if (error.status !== 401 || isOpen) {
        return throwError(() => error);
      }

      // Access token expired / invalid — try to refresh and retry once.
      const refresh = authService.getRefreshToken();
      if (!refresh) {
        authService.logout();
        return throwError(() => error);
      }

      return authService.refreshToken().pipe(
        switchMap(({ access }) => next(attachToken(req, access))),
        catchError(refreshErr => {
          // Refresh token also rejected — session is unrecoverable.
          authService.logout();
          return throwError(() => refreshErr);
        })
      );
    })
  );
};

function attachToken(req: HttpRequest<unknown>, token: string | null): HttpRequest<unknown> {
  if (!token) return req;
  return req.clone({ setHeaders: { Authorization: `Bearer ${token}` } });
}

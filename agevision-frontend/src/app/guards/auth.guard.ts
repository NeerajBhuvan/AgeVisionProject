import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '../services/auth.service';

/**
 * Functional route guard that redirects unauthenticated
 * users to /login while preserving the intended URL.
 */
export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  if (authService.getAccessToken()) {
    return true;
  }

  // Redirect to login and remember where the user wanted to go
  router.navigate(['/login'], {
    queryParams: { returnUrl: state.url }
  });
  return false;
};

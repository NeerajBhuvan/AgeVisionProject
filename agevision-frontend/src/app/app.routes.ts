import { Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: 'login',
    loadComponent: () =>
      import('./pages/auth/login/login.component')
        .then(m => m.LoginComponent),
    data: { animation: 'Login' }
  },
  {
    path: 'register',
    loadComponent: () =>
      import('./pages/auth/register/register.component')
        .then(m => m.RegisterComponent),
    data: { animation: 'Register' }
  },
  {
    path: 'forgot-password',
    loadComponent: () =>
      import('./pages/auth/forgot-password/forgot-password.component')
        .then(m => m.ForgotPasswordComponent),
    data: { animation: 'ForgotPassword' }
  },
  {
    path: 'dashboard',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component')
        .then(m => m.DashboardComponent),
    data: { animation: 'Dashboard' }
  },
  {
    path: 'predict',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/predict/predict.component')
        .then(m => m.PredictComponent),
    data: { animation: 'Predict' }
  },
  {
    path: 'progress',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/progress/progress.component')
        .then(m => m.ProgressComponent),
    data: { animation: 'Progress' }
  },
  {
    path: 'history',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/history/history.component')
        .then(m => m.HistoryComponent),
    data: { animation: 'History' }
  },
  {
    path: 'analytics',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/analytics/analytics.component')
        .then(m => m.AnalyticsComponent),
    data: { animation: 'Analytics' }
  },
  {
    path: 'settings',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/settings/settings.component')
        .then(m => m.SettingsComponent),
    data: { animation: 'Settings' }
  },
  {
    path: '**',
    redirectTo: 'dashboard'
  }
];

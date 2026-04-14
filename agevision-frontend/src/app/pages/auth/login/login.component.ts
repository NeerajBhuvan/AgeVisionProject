import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router, RouterLink, ActivatedRoute } from '@angular/router';
import { AuthService } from '../../../services/auth.service';
import { LogoComponent } from '../../../shared/logo/logo.component';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink, LogoComponent],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  loginForm: FormGroup;
  isLoading = false;
  errorMessage = '';
  isSuspended = false;
  showPassword = false;
  private returnUrl = '/dashboard';

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router,
    private route: ActivatedRoute
  ) {
    this.loginForm = this.fb.group({
      username: ['', [Validators.required]],
      password: ['', [Validators.required, Validators.minLength(8)]],
      rememberMe: [false]
    });

    // Capture the returnUrl from query params (set by authGuard).
    // Only allow internal, same-origin paths to prevent open-redirect attacks
    // via a crafted ?returnUrl=https://evil.example link.
    const requested = this.route.snapshot.queryParams['returnUrl'];
    this.returnUrl = LoginComponent.sanitizeReturnUrl(requested);
  }

  /**
   * Accept only internal paths beginning with a single "/" and not "//"
   * (protocol-relative) or "/\\" (backslash trick). Anything else — absolute
   * URLs, javascript:, data:, empty — falls back to /dashboard.
   */
  private static sanitizeReturnUrl(url: unknown): string {
    if (typeof url !== 'string' || url.length === 0) return '/dashboard';
    if (!url.startsWith('/')) return '/dashboard';
    if (url.startsWith('//') || url.startsWith('/\\')) return '/dashboard';
    return url;
  }

  /** Submit login form */
  onSubmit(): void {
    // Mark all controls as touched to show validation
    this.loginForm.markAllAsTouched();

    if (this.loginForm.invalid) {
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';
    this.isSuspended = false;

    const { username, password } = this.loginForm.value;

    this.authService.login({ username, password }).subscribe({
      next: () => {
        this.isLoading = false;
        this.router.navigate([this.returnUrl]);
      },
      error: (err) => {
        this.isLoading = false;
        this.isSuspended = !!err.error?.account_suspended;
        this.errorMessage =
          err.error?.detail ||
          err.error?.error ||
          err.error?.non_field_errors?.[0] ||
          'Invalid credentials. Please try again.';
      }
    });
  }

  /** Toggle password visibility */
  togglePasswordVisibility(): void {
    this.showPassword = !this.showPassword;
  }

  /** Navigate to register page */
  navigateToRegister(): void {
    this.router.navigate(['/register']);
  }

  /** Form control getters for template convenience */
  get f() {
    return this.loginForm.controls;
  }
}

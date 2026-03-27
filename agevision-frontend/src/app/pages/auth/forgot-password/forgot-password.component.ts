import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService, ForgotPasswordResponse } from '../../../services/auth.service';
import { LogoComponent } from '../../../shared/logo/logo.component';

@Component({
  selector: 'app-forgot-password',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink, LogoComponent],
  templateUrl: './forgot-password.component.html',
  styleUrl: './forgot-password.component.scss'
})
export class ForgotPasswordComponent {
  emailForm: FormGroup;
  resetForm: FormGroup;
  isLoading = false;
  errorMessage = '';
  successMessage = '';

  // Recovery state
  step: 'email' | 'hint' | 'reset' | 'done' = 'email';
  passwordHint = '';
  passwordLength = 0;
  recoveryToken = '';
  recoveredEmail = '';
  recoveredUsername = '';
  recoveredPassword = '';
  showRecoveredPassword = false;
  showNewPassword = false;
  showConfirmPassword = false;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router
  ) {
    this.emailForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]]
    });

    this.resetForm = this.fb.group({
      newPassword: ['', [Validators.required, Validators.minLength(8)]],
      confirmPassword: ['', [Validators.required]]
    });
  }

  /** Step 1: Submit email to get recovery hint */
  onSubmitEmail(): void {
    this.emailForm.markAllAsTouched();
    if (this.emailForm.invalid) return;

    this.isLoading = true;
    this.errorMessage = '';
    this.successMessage = '';
    const email = this.emailForm.value.email;

    this.authService.forgotPassword(email).subscribe({
      next: (res: ForgotPasswordResponse) => {
        this.isLoading = false;
        this.recoveredEmail = email;

        if (res.recovery_available) {
          this.passwordHint = res.password_hint || '';
          this.passwordLength = res.password_length || 0;
          this.recoveryToken = res.recovery_token || '';
          this.recoveredUsername = res.username || '';
          this.step = 'hint';
        } else {
          this.successMessage = res.message;
        }
      },
      error: (err) => {
        this.isLoading = false;
        this.errorMessage = err.error?.error || 'Something went wrong. Please try again.';
      }
    });
  }

  /** Step 2a: Show full recovered password */
  onRevealPassword(): void {
    this.isLoading = true;
    this.errorMessage = '';

    this.authService.recoverPassword(this.recoveredEmail, this.recoveryToken).subscribe({
      next: (res) => {
        this.isLoading = false;
        if (res.password) {
          this.recoveredPassword = res.password;
        } else {
          this.errorMessage = 'Could not recover password.';
        }
      },
      error: (err) => {
        this.isLoading = false;
        this.errorMessage = err.error?.error || 'Recovery failed. Token may have expired.';
      }
    });
  }

  /** Step 2b: Switch to reset form */
  onGoToReset(): void {
    this.step = 'reset';
    this.errorMessage = '';
  }

  /** Step 3: Submit new password */
  onResetPassword(): void {
    this.resetForm.markAllAsTouched();
    if (this.resetForm.invalid) return;

    const { newPassword, confirmPassword } = this.resetForm.value;
    if (newPassword !== confirmPassword) {
      this.errorMessage = 'Passwords do not match';
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';

    this.authService.resetPassword(this.recoveredEmail, this.recoveryToken, newPassword).subscribe({
      next: (res) => {
        this.isLoading = false;
        this.successMessage = res.message;
        this.step = 'done';
      },
      error: (err) => {
        this.isLoading = false;
        this.errorMessage = err.error?.error || 'Reset failed. Please try again.';
      }
    });
  }

  goToLogin(): void {
    this.router.navigate(['/login']);
  }

  get ef() { return this.emailForm.controls; }
  get rf() { return this.resetForm.controls; }
}

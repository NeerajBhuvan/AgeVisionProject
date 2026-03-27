import { Component, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ReactiveFormsModule,
  FormBuilder,
  FormGroup,
  Validators,
  AbstractControl,
  ValidationErrors
} from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthService } from '../../../services/auth.service';
import { LogoComponent } from '../../../shared/logo/logo.component';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink, LogoComponent],
  templateUrl: './register.component.html',
  styleUrl: './register.component.scss'
})
export class RegisterComponent implements OnDestroy {
  registerForm: FormGroup;
  isLoading = false;
  errorMessage = '';
  showPassword = false;
  showConfirmPassword = false;
  passwordStrength = 0; // 0–4

  private pwSub!: Subscription;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router
  ) {
    this.registerForm = this.fb.group(
      {
        first_name: ['', [Validators.required]],
        last_name: ['', [Validators.required]],
        email: ['', [Validators.required, Validators.email]],
        username: [
          '',
          [
            Validators.required,
            Validators.minLength(4),
            Validators.pattern(/^[a-zA-Z0-9_]+$/)
          ]
        ],
        password: ['', [Validators.required, Validators.minLength(8)]],
        confirmPassword: ['', [Validators.required]],
        terms: [false, [Validators.requiredTrue]]
      },
      { validators: RegisterComponent.passwordMatchValidator }
    );

    // Recalculate strength whenever password changes
    this.pwSub = this.registerForm
      .get('password')!
      .valueChanges.subscribe((pw: string) => {
        this.passwordStrength = this.calculatePasswordStrength(pw);
      });
  }

  ngOnDestroy(): void {
    this.pwSub?.unsubscribe();
  }

  /* ── Custom validator: passwords must match ── */
  static passwordMatchValidator(group: AbstractControl): ValidationErrors | null {
    const pw = group.get('password')?.value;
    const cpw = group.get('confirmPassword')?.value;
    if (pw && cpw && pw !== cpw) {
      group.get('confirmPassword')?.setErrors({ passwordMismatch: true });
      return { passwordMismatch: true };
    }
    return null;
  }

  /* ── Password strength: 0 (empty) → 4 (strong) ── */
  calculatePasswordStrength(password: string): number {
    if (!password) return 0;
    let score = 0;
    if (password.length >= 8) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[a-z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;
    // Normalize to 1-4 range (five checks → max 5, cap at 4)
    return Math.min(score, 4);
  }

  get strengthLabel(): string {
    switch (this.passwordStrength) {
      case 1: return 'Weak';
      case 2: return 'Fair';
      case 3: return 'Good';
      case 4: return 'Strong';
      default: return '';
    }
  }

  get strengthClass(): string {
    switch (this.passwordStrength) {
      case 1: return 'weak';
      case 2: return 'fair';
      case 3: return 'good';
      case 4: return 'strong';
      default: return '';
    }
  }

  /* ── Submit ── */
  onSubmit(): void {
    this.registerForm.markAllAsTouched();

    if (this.registerForm.invalid) {
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';

    const { first_name, last_name, email, username, password } =
      this.registerForm.value;

    this.authService
      .register({ first_name, last_name, email, username, password })
      .subscribe({
        next: () => {
          this.isLoading = false;
          this.router.navigate(['/dashboard']);
        },
        error: (err) => {
          this.isLoading = false;
          // Django may return field-level errors as {field: ["msg"]}
          if (err.error && typeof err.error === 'object') {
            const messages: string[] = [];
            for (const key of Object.keys(err.error)) {
              const val = err.error[key];
              if (Array.isArray(val)) {
                messages.push(...val);
              } else if (typeof val === 'string') {
                messages.push(val);
              }
            }
            this.errorMessage =
              messages.join(' ') || 'Registration failed. Please try again.';
          } else {
            this.errorMessage = 'Registration failed. Please try again.';
          }
        }
      });
  }

  togglePasswordVisibility(): void {
    this.showPassword = !this.showPassword;
  }

  toggleConfirmPasswordVisibility(): void {
    this.showConfirmPassword = !this.showConfirmPassword;
  }

  navigateToLogin(): void {
    this.router.navigate(['/login']);
  }

  get f() {
    return this.registerForm.controls;
  }
}

import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-logo',
  standalone: true,
  template: `
    <svg [attr.width]="width" [attr.height]="height" viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
      <!-- Outer eye shape -->
      <path d="M60 10 C30 10, 5 40, 5 40 C5 40, 30 70, 60 70 C90 70, 115 40, 115 40 C115 40, 90 10, 60 10Z"
            fill="none" class="logo-eye" stroke-width="3.5"/>
      <!-- Circuit lines left -->
      <line x1="5" y1="40" x2="0" y2="40" class="logo-circuit" stroke-width="2.5"/>
      <line x1="10" y1="28" x2="2" y2="22" class="logo-circuit" stroke-width="2"/>
      <line x1="10" y1="52" x2="2" y2="58" class="logo-circuit" stroke-width="2"/>
      <!-- Circuit lines right -->
      <line x1="115" y1="40" x2="120" y2="40" class="logo-circuit" stroke-width="2.5"/>
      <line x1="110" y1="28" x2="118" y2="22" class="logo-circuit" stroke-width="2"/>
      <line x1="110" y1="52" x2="118" y2="58" class="logo-circuit" stroke-width="2"/>
      <!-- Outer ring (gold, dashed) -->
      <circle cx="60" cy="40" r="22" fill="none" class="logo-circuit" stroke-width="3" stroke-dasharray="18 6"/>
      <!-- Middle ring -->
      <circle cx="60" cy="40" r="15" fill="none" class="logo-eye" stroke-width="2.5"/>
      <!-- Inner ring -->
      <circle cx="60" cy="40" r="9" fill="none" class="logo-eye" stroke-width="2"/>
      <!-- Pupil -->
      <circle cx="60" cy="40" r="4" class="logo-pupil"/>
      <!-- Highlight -->
      <circle cx="57" cy="37" r="1.5" fill="#fff" opacity="0.8"/>
    </svg>
  `,
  styles: [`
    :host { display: inline-flex; align-items: center; justify-content: center; }
    .logo-eye { stroke: var(--logo-primary); }
    .logo-circuit { stroke: var(--logo-accent); }
    .logo-pupil { fill: var(--logo-primary); }
  `]
})
export class LogoComponent {
  @Input() width = '80';
  @Input() height = '54';
}

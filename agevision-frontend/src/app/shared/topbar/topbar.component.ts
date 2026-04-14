import {
  Component,
  OnInit,
  OnDestroy,
  HostListener,
  ElementRef,
  signal
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, NavigationEnd } from '@angular/router';
import { Subscription, filter } from 'rxjs';
import {
  trigger,
  transition,
  style,
  animate
} from '@angular/animations';
import { ThemeService, Theme } from '../../services/theme.service';
import { AuthService, User } from '../../services/auth.service';
import { TimezoneService } from '../../services/timezone.service';
import { NotificationService } from '../../services/notification.service';
import { AppNotification } from '../../models/notification';

interface PageMeta {
  title: string;
  subtitle: string;
}

@Component({
  selector: 'app-topbar',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './topbar.component.html',
  styleUrl: './topbar.component.scss',
  animations: [
    trigger('fadeSlide', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-8px) scale(0.96)' }),
        animate('180ms cubic-bezier(0.4, 0, 0.2, 1)',
          style({ opacity: 1, transform: 'translateY(0) scale(1)' }))
      ]),
      transition(':leave', [
        animate('120ms cubic-bezier(0.4, 0, 0.2, 1)',
          style({ opacity: 0, transform: 'translateY(-8px) scale(0.96)' }))
      ])
    ])
  ]
})
export class TopbarComponent implements OnInit, OnDestroy {
  pageTitle = signal<PageMeta>({ title: 'Dashboard', subtitle: 'Overview' });
  avatarDropdownOpen = signal(false);
  notifPanelOpen = signal(false);
  showLogoutModal = false;
  currentTheme: Theme = 'dark';
  currentUser: User | null = null;
  currentTime = '';
  activeToast: AppNotification | null = null;

  private routeSub!: Subscription;
  private themeSub!: Subscription;
  private userSub!: Subscription;
  private toastSub!: Subscription;
  private clockInterval: any;
  private toastTimeout: any;

  private readonly pageTitles: Record<string, PageMeta> = {
    '/dashboard':  { title: 'Dashboard',      subtitle: 'Overview' },
    '/predict':    { title: 'Age Predict',     subtitle: 'Upload Image' },
    '/batch-predict': { title: 'Batch Predict', subtitle: 'Multi-Image Run' },
    '/progress':   { title: 'Age Progress',    subtitle: 'Simulate Aging' },
    '/history':    { title: 'History',          subtitle: 'Past Analyses' },
    '/analytics':  { title: 'Analytics',        subtitle: 'Insights & Trends' },
    '/settings':   { title: 'Settings',         subtitle: 'Preferences' },
    '/admin':      { title: 'Admin Panel',       subtitle: 'Platform Monitoring' },
  };

  constructor(
    private router: Router,
    private elRef: ElementRef,
    public themeService: ThemeService,
    public authService: AuthService,
    public tzService: TimezoneService,
    public notifService: NotificationService
  ) {}

  ngOnInit(): void {
    this.updatePageTitle(this.router.url);
    this.routeSub = this.router.events
      .pipe(filter(event => event instanceof NavigationEnd))
      .subscribe(event => {
        this.updatePageTitle((event as NavigationEnd).urlAfterRedirects);
      });
    this.themeSub = this.themeService.currentTheme$.subscribe(theme => {
      this.currentTheme = theme;
    });
    this.userSub = this.authService.currentUser$.subscribe(user => {
      this.currentUser = user;
    });
    this.updateClock();
    this.clockInterval = setInterval(() => this.updateClock(), 1000);

    this.toastSub = this.notifService.toast$.subscribe(toast => {
      if (toast) {
        this.activeToast = toast;
        if (this.toastTimeout) clearTimeout(this.toastTimeout);
        this.toastTimeout = setTimeout(() => {
          this.activeToast = null;
          this.notifService.dismissToast();
        }, 4000);
      }
    });
  }

  ngOnDestroy(): void {
    this.routeSub?.unsubscribe();
    this.themeSub?.unsubscribe();
    this.userSub?.unsubscribe();
    this.toastSub?.unsubscribe();
    if (this.clockInterval) clearInterval(this.clockInterval);
    if (this.toastTimeout) clearTimeout(this.toastTimeout);
  }

  private updateClock(): void {
    const now = new Date();
    const tz = this.tzService.timezone;
    try {
      const date = now.toLocaleDateString('en-US', {
        timeZone: tz,
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
      const time = now.toLocaleTimeString('en-US', {
        timeZone: tz,
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
      });
      this.currentTime = `${date}, ${time} ${this.tzService.getTimezoneAbbr()}`;
    } catch {
      this.currentTime = now.toLocaleString();
    }
  }

  private updatePageTitle(url: string): void {
    const path = url.split('?')[0].split('#')[0];
    const meta = this.pageTitles[path];
    this.pageTitle.set(meta ?? { title: 'AgeVision', subtitle: 'AI' });
  }

  toggleTheme(): void {
    this.themeService.toggleTheme();
  }

  toggleAvatarDropdown(): void {
    this.avatarDropdownOpen.update(v => !v);
    this.notifPanelOpen.set(false);
  }

  toggleNotifPanel(): void {
    this.notifPanelOpen.update(v => !v);
    this.avatarDropdownOpen.set(false);
  }

  onNotifClick(n: AppNotification): void {
    this.notifService.markAsRead(n.id);
    if (n.route) {
      this.router.navigate([n.route]);
      this.notifPanelOpen.set(false);
    }
  }

  removeNotif(event: MouseEvent, id: string): void {
    event.stopPropagation();
    this.notifService.remove(id);
  }

  dismissToast(): void {
    this.activeToast = null;
    if (this.toastTimeout) clearTimeout(this.toastTimeout);
    this.notifService.dismissToast();
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    if (this.avatarDropdownOpen()) {
      const dropdown = this.elRef.nativeElement.querySelector('.topbar-avatar-wrap');
      if (dropdown && !dropdown.contains(target)) {
        this.avatarDropdownOpen.set(false);
      }
    }
    if (this.notifPanelOpen()) {
      const notifWrap = this.elRef.nativeElement.querySelector('.notif-wrap');
      if (notifWrap && !notifWrap.contains(target)) {
        this.notifPanelOpen.set(false);
      }
    }
  }

  openLogoutModal(): void {
    this.avatarDropdownOpen.set(false);
    this.showLogoutModal = true;
  }

  closeLogoutModal(): void {
    this.showLogoutModal = false;
  }

  confirmLogout(): void {
    this.showLogoutModal = false;
    this.authService.logout();
  }

  isUserActive(): boolean {
    return this.currentUser?.is_active ?? false;
  }

  navigateToSettings(section: string): void {
    this.avatarDropdownOpen.set(false);
    this.router.navigate(['/settings'], { queryParams: { section } });
  }
}

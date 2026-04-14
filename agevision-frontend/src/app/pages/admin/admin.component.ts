import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';

interface AdminUserRow {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
  is_superuser: boolean;
  is_staff: boolean;
  plan: string;
  avatar_url: string;
  date_joined: string | null;
  last_login: string | null;
  prediction_count: number;
  progression_count: number;
}

interface BreakdownItem { label: string; count: number; color: string; }

@Component({
  selector: 'app-admin',
  imports: [CommonModule, FormsModule],
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.scss'
})
export class AdminComponent implements OnInit, OnDestroy {
  loading = true;
  loadingUsers = false;
  loadingHealth = false;

  // Dashboard KPIs
  totalUsers = 0;
  activeUsers = 0;
  totalPredictions = 0;
  totalProgressions = 0;
  avgAge = 0;
  avgConfidence = 0;
  avgPredictionTimeMs = 0;
  avgProgressionTimeMs = 0;

  // Breakdowns
  detectorBreakdown: BreakdownItem[] = [];
  ganModelBreakdown: BreakdownItem[] = [];
  totalDetectorCount = 0;
  totalGanModelCount = 0;

  // Daily activity (last 14 days)
  predictionDailyCounts: { date: string; count: number }[] = [];
  progressionDailyCounts: { date: string; count: number }[] = [];

  // Users table
  users: AdminUserRow[] = [];
  userPage = 1;
  userLimit = 25;
  userTotal = 0;
  userPages = 1;
  userSearch = '';
  private searchTimer: any = null;

  // Suspend confirmation modal
  showSuspendModal = false;
  pendingSuspendUser: AdminUserRow | null = null;

  // System health
  mongodbStatus: 'connected' | 'error' | 'unknown' = 'unknown';
  mongodbError: string | null = null;
  checkpointFiles: { name: string; size_bytes: number }[] = [];
  totalCheckpointBytes = 0;
  uploadsBytes = 0;
  gpu: {
    available: boolean;
    name: string | null;
    memory_total_bytes: number;
    memory_allocated_bytes: number;
  } = { available: false, name: null, memory_total_bytes: 0, memory_allocated_bytes: 0 };

  private healthTimer: any = null;

  private readonly detectorColors: Record<string, string> = {
    mivolo: '#7f5af0',
    insightface: '#2cb67d',
    deepface: '#ff8906',
    opencv: '#9ca3af',
    unknown: '#6b7280'
  };

  private readonly ganColors: Record<string, string> = {
    sam: '#7f5af0',
    'sam-ffhq': '#7f5af0',
    'sam-indian': '#ef4444',
    'fast-aginggan': '#2cb67d',
    fast_aging: '#2cb67d',
    hrfae: '#ff8906',
    unknown: '#9ca3af'
  };

  constructor(
    private api: ApiService,
    private notify: NotificationService
  ) {}

  ngOnInit(): void {
    this.loadDashboard();
    this.loadUsers();
    this.loadHealth();
    // Auto-refresh health every 30s
    this.healthTimer = setInterval(() => this.loadHealth(), 30000);
  }

  ngOnDestroy(): void {
    if (this.healthTimer) clearInterval(this.healthTimer);
    if (this.searchTimer) clearTimeout(this.searchTimer);
  }

  /* ── Dashboard ── */
  loadDashboard(): void {
    this.loading = true;
    this.api.getAdminDashboard().subscribe({
      next: (data) => {
        this.totalUsers = data.total_users || 0;
        this.activeUsers = data.active_users || 0;
        this.totalPredictions = data.total_predictions || 0;
        this.totalProgressions = data.total_progressions || 0;
        this.avgAge = data.average_predicted_age || 0;
        this.avgConfidence = data.average_confidence || 0;
        this.avgPredictionTimeMs = data.avg_prediction_time_ms || 0;
        this.avgProgressionTimeMs = data.avg_progression_time_ms || 0;

        const detectors = data.detector_breakdown || [];
        this.totalDetectorCount = detectors.reduce((s: number, d: any) => s + d.count, 0);
        this.detectorBreakdown = detectors.map((d: any) => ({
          label: d.detector || 'unknown',
          count: d.count,
          color: this.detectorColors[(d.detector || '').toLowerCase()] || '#9ca3af'
        }));

        const gans = data.gan_model_breakdown || [];
        this.totalGanModelCount = gans.reduce((s: number, g: any) => s + g.count, 0);
        this.ganModelBreakdown = gans.map((g: any) => ({
          label: g.model || 'unknown',
          count: g.count,
          color: this.ganColors[(g.model || '').toLowerCase()] || '#9ca3af'
        }));

        this.predictionDailyCounts = data.prediction_daily_counts || [];
        this.progressionDailyCounts = data.progression_daily_counts || [];

        this.loading = false;
      },
      error: (err) => {
        this.loading = false;
        this.notify.error('Admin dashboard', err?.error?.detail || 'Failed to load dashboard data');
      }
    });
  }

  /* ── Users ── */
  loadUsers(): void {
    this.loadingUsers = true;
    this.api.getAdminUsers(this.userPage, this.userLimit, this.userSearch).subscribe({
      next: (data) => {
        this.users = data.users || [];
        this.userTotal = data.total || 0;
        this.userPages = data.pages || 1;
        this.loadingUsers = false;
      },
      error: (err) => {
        this.loadingUsers = false;
        this.notify.error('Users', err?.error?.detail || 'Failed to load users');
      }
    });
  }

  onSearchChange(): void {
    if (this.searchTimer) clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => {
      this.userPage = 1;
      this.loadUsers();
    }, 350);
  }

  goToPage(page: number): void {
    if (page < 1 || page > this.userPages) return;
    this.userPage = page;
    this.loadUsers();
  }

  openSuspendModal(user: AdminUserRow): void {
    this.pendingSuspendUser = user;
    this.showSuspendModal = true;
  }

  closeSuspendModal(): void {
    this.showSuspendModal = false;
    this.pendingSuspendUser = null;
  }

  confirmSuspend(): void {
    const user = this.pendingSuspendUser;
    if (!user) {
      this.closeSuspendModal();
      return;
    }
    this.showSuspendModal = false;
    this.api.suspendAdminUser(user.id).subscribe({
      next: () => {
        user.is_active = false;
        this.notify.success('User suspended', `${user.username} can no longer log in.`);
        this.pendingSuspendUser = null;
      },
      error: (err) => {
        this.notify.error('Suspend failed', err?.error?.error || 'Could not suspend user.');
        this.pendingSuspendUser = null;
      }
    });
  }

  reinstateUser(user: AdminUserRow): void {
    this.api.reinstateAdminUser(user.id).subscribe({
      next: () => {
        user.is_active = true;
        this.notify.success('User reinstated', `${user.username} can log in again.`);
      },
      error: (err) => {
        this.notify.error('Reinstate failed', err?.error?.error || 'Could not reinstate user.');
      }
    });
  }

  /* ── Health ── */
  loadHealth(): void {
    this.loadingHealth = true;
    this.api.getAdminSystemHealth().subscribe({
      next: (data) => {
        this.mongodbStatus = data?.mongodb?.status || 'unknown';
        this.mongodbError = data?.mongodb?.error || null;
        this.checkpointFiles = data?.checkpoints?.files || [];
        this.totalCheckpointBytes = data?.checkpoints?.total_bytes || 0;
        this.uploadsBytes = data?.uploads?.total_bytes || 0;
        this.gpu = data?.gpu || this.gpu;
        this.loadingHealth = false;
      },
      error: () => {
        this.loadingHealth = false;
      }
    });
  }

  /* ── Helpers ── */
  getPct(count: number, total: number): number {
    return total > 0 ? Math.round((count / total) * 100) : 0;
  }

  getMaxDailyCount(series: { date: string; count: number }[]): number {
    if (!series.length) return 1;
    return Math.max(...series.map(d => d.count), 1);
  }

  formatBytes(bytes: number): string {
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit++;
    }
    return value.toFixed(value < 10 ? 2 : 1) + ' ' + units[unit];
  }

  gpuMemPct(): number {
    if (!this.gpu.memory_total_bytes) return 0;
    return Math.round((this.gpu.memory_allocated_bytes / this.gpu.memory_total_bytes) * 100);
  }

  shortDate(iso: string | null): string {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    } catch {
      return '—';
    }
  }

  get pageNumbers(): number[] {
    const max = 5;
    const pages: number[] = [];
    let start = Math.max(1, this.userPage - 2);
    let end = Math.min(this.userPages, start + max - 1);
    start = Math.max(1, end - max + 1);
    for (let i = start; i <= end; i++) pages.push(i);
    return pages;
  }
}

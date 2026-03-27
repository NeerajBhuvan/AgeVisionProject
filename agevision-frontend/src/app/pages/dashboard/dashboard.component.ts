import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { AuthService } from '../../services/auth.service';
import { TimezoneService } from '../../services/timezone.service';

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent implements OnInit {
  loading = true;
  userName = '';

  totalPredictions = 0;
  totalProgressions = 0;
  totalAnalyses = 0;
  weekPredictions = 0;
  weekProgressions = 0;
  avgConfidence = 0;
  avgProcessingTime = 0;

  recentActivities: any[] = [];
  dailyCounts: { date: string; count: number }[] = [];

  constructor(
    private api: ApiService,
    private auth: AuthService,
    private tz: TimezoneService
  ) {}

  ngOnInit(): void {
    this.userName = this.auth.getDisplayName();
    this.loadDashboardData();
  }

  loadDashboardData(): void {
    this.loading = true;
    let loaded = 0;
    const checkDone = () => { if (++loaded >= 2) this.loading = false; };

    this.api.getAnalytics().subscribe({
      next: (data) => {
        this.totalPredictions = data.total_predictions || 0;
        this.totalProgressions = data.total_progressions || 0;
        this.totalAnalyses = this.totalPredictions + this.totalProgressions;
        this.weekPredictions = data.week_predictions || 0;
        this.weekProgressions = data.week_progressions || 0;
        this.avgConfidence = data.average_confidence || 0;
        this.dailyCounts = (data.daily_counts || []).reverse();
        checkDone();
      },
      error: () => checkDone()
    });

    this.api.getHistory().subscribe({
      next: (data: any) => {
        const predictions = (data.predictions || []).map((p: any) => ({
          id: p.id,
          type: 'prediction' as const,
          icon: p.face_count > 1 ? 'fa-solid fa-users' : 'fa-solid fa-user',
          title: p.face_count > 1 ? 'Group Photo Analysis' : 'Portrait Analysis',
          age: String(p.predicted_age),
          confidence: p.confidence,
          model: p.detector_used || 'ensemble',
          date: new Date(p.created_at),
          processingTime: p.processing_time_ms
        }));

        const progressions = (data.progressions || []).map((p: any) => ({
          id: p.id,
          type: 'progression' as const,
          icon: 'fa-solid fa-masks-theater',
          title: 'Age Progression',
          age: `${p.current_age} \u2192 ${p.target_age}`,
          confidence: null,
          model: p.model_used || 'SAM',
          date: new Date(p.created_at),
          processingTime: p.processing_time_ms
        }));

        const all = [...predictions, ...progressions];
        all.sort((a, b) => b.date.getTime() - a.date.getTime());
        this.recentActivities = all.slice(0, 5);

        if (all.length > 0) {
          const totalMs = all.reduce((sum: number, a: any) => sum + (a.processingTime || 0), 0);
          this.avgProcessingTime = totalMs / all.length;
        }

        checkDone();
      },
      error: () => checkDone()
    });
  }

  getTimeAgo(date: Date): string {
    return this.tz.formatTimeAgo(date);
  }

  formatProcessingTime(ms: number): string {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  getMaxDailyCount(): number {
    if (!this.dailyCounts.length) return 1;
    return Math.max(...this.dailyCounts.map(d => d.count), 1);
  }

  getDayLabel(dateStr: string): string {
    return this.tz.getDayLabel(dateStr);
  }
}

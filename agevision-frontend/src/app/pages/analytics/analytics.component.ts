import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { TimezoneService } from '../../services/timezone.service';

@Component({
  selector: 'app-analytics',
  imports: [CommonModule],
  templateUrl: './analytics.component.html',
  styleUrl: './analytics.component.scss'
})
export class AnalyticsComponent implements OnInit {
  loading = true;

  // KPIs
  totalPredictions = 0;
  totalProgressions = 0;
  totalAnalyses = 0;
  weekPredictions = 0;
  weekProgressions = 0;
  avgAge = 0;
  avgConfidence = 0;

  // Weekly bar chart
  dailyCounts: { date: string; count: number }[] = [];

  // Gender distribution
  genderDist: { gender: string; count: number; color: string }[] = [];
  totalGenderCount = 0;

  // Emotion distribution
  emotionDist: { emotion: string; count: number; color: string }[] = [];
  totalEmotionCount = 0;

  // Age distribution (computed from history predictions)
  ageDist: { label: string; count: number; pct: number; color: string }[] = [];
  ageDonutSegments: { label: string; pct: number; color: string; offset: number }[] = [];

  // Recent activity timeline
  recentTimeline: { title: string; meta: string; icon: string }[] = [];

  private readonly emotionColors: Record<string, string> = {
    happy: '#2cb67d',
    sad: '#3b82f6',
    angry: '#ef4444',
    surprise: '#ff8906',
    fear: '#8b5cf6',
    disgust: '#6b7280',
    neutral: '#7f5af0',
    unknown: '#9ca3af'
  };

  private readonly genderColors: Record<string, string> = {
    man: '#7f5af0',
    woman: '#2cb67d',
    male: '#7f5af0',
    female: '#2cb67d',
    unknown: '#9ca3af'
  };

  private readonly ageColors = ['#2cb67d', '#7f5af0', '#ff8906', '#5b21b6', '#ef4444'];

  constructor(
    private api: ApiService,
    private tz: TimezoneService
  ) {}

  ngOnInit(): void {
    this.loadAnalytics();
  }

  loadAnalytics(): void {
    this.loading = true;
    let loaded = 0;
    const checkDone = () => { if (++loaded >= 2) this.loading = false; };

    // Load analytics endpoint
    this.api.getAnalytics().subscribe({
      next: (data) => {
        this.totalPredictions = data.total_predictions || 0;
        this.totalProgressions = data.total_progressions || 0;
        this.totalAnalyses = this.totalPredictions + this.totalProgressions;
        this.weekPredictions = data.week_predictions || 0;
        this.weekProgressions = data.week_progressions || 0;
        this.avgAge = data.average_predicted_age || 0;
        this.avgConfidence = data.average_confidence || 0;

        // Daily counts (reverse to get oldest first for chart)
        this.dailyCounts = (data.daily_counts || []).reverse();

        // Gender distribution
        const genders = data.gender_distribution || [];
        this.totalGenderCount = genders.reduce((s: number, g: any) => s + g.count, 0);
        this.genderDist = genders.map((g: any) => ({
          gender: g.gender || 'Unknown',
          count: g.count,
          color: this.genderColors[(g.gender || '').toLowerCase()] || '#9ca3af'
        }));

        // Emotion distribution
        const emotions = data.emotion_distribution || [];
        this.totalEmotionCount = emotions.reduce((s: number, e: any) => s + e.count, 0);
        this.emotionDist = emotions.map((e: any) => ({
          emotion: e.emotion || 'Unknown',
          count: e.count,
          color: this.emotionColors[(e.emotion || '').toLowerCase()] || '#9ca3af'
        }));

        checkDone();
      },
      error: () => checkDone()
    });

    // Load history for age distribution and timeline
    this.api.getHistory().subscribe({
      next: (data: any) => {
        const predictions = data.predictions || [];
        const progressions = data.progressions || [];

        // Compute age distribution from actual predicted ages
        const ageBuckets = [
          { label: '0-17', min: 0, max: 17 },
          { label: '18-25', min: 18, max: 25 },
          { label: '26-35', min: 26, max: 35 },
          { label: '36-50', min: 36, max: 50 },
          { label: '50+', min: 51, max: 999 }
        ];

        const totalPreds = predictions.length;
        const bucketCounts = ageBuckets.map((bucket, i) => {
          const count = predictions.filter((p: any) =>
            p.predicted_age >= bucket.min && p.predicted_age <= bucket.max
          ).length;
          const pct = totalPreds > 0 ? Math.round((count / totalPreds) * 100) : 0;
          return {
            label: bucket.label,
            count,
            pct,
            color: this.ageColors[i]
          };
        });

        this.ageDist = bucketCounts.filter(b => b.count > 0);

        // Build donut segments
        let offset = 0;
        this.ageDonutSegments = this.ageDist.map(d => {
          const seg = { label: d.label, pct: d.pct, color: d.color, offset };
          offset += d.pct;
          return seg;
        });

        // Build recent timeline from all records
        const allRecords = [
          ...predictions.map((p: any) => ({
            title: `Prediction: Age ${p.predicted_age}`,
            meta: this.tz.formatTimeAgo(new Date(p.created_at)) + ' \u2022 ' + (p.detector_used || 'ensemble'),
            icon: p.face_count > 1 ? 'fa-solid fa-users' : 'fa-solid fa-user',
            date: new Date(p.created_at)
          })),
          ...progressions.map((p: any) => ({
            title: `Progression: ${p.current_age} \u2192 ${p.target_age}`,
            meta: this.tz.formatTimeAgo(new Date(p.created_at)) + ' \u2022 ' + (p.model_used || 'SAM'),
            icon: 'fa-solid fa-masks-theater',
            date: new Date(p.created_at)
          }))
        ];
        allRecords.sort((a: any, b: any) => b.date.getTime() - a.date.getTime());
        this.recentTimeline = allRecords.slice(0, 5);

        checkDone();
      },
      error: () => checkDone()
    });
  }

  getMaxDailyCount(): number {
    if (!this.dailyCounts.length) return 1;
    return Math.max(...this.dailyCounts.map(d => d.count), 1);
  }

  getDayLabel(dateStr: string): string {
    return this.tz.getDayLabel(dateStr);
  }

  getPct(count: number, total: number): number {
    return total > 0 ? Math.round((count / total) * 100) : 0;
  }
}

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { TimezoneService } from '../../services/timezone.service';
import { TimeStats, ModelPerformance } from '../../models/analytics';

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

  // Confidence distribution
  confidenceDist: { range: string; count: number; color: string }[] = [];
  totalConfCount = 0;

  // Model performance
  modelPerf: (ModelPerformance & { color: string })[] = [];

  // Model/Detector distribution
  detectorDist: { detector: string; count: number; color: string }[] = [];
  totalDetectorCount = 0;
  modelDist: { model: string; count: number; color: string }[] = [];
  totalModelCount = 0;

  // Processing time stats
  predTimeStats: TimeStats = { avg: 0, min: 0, max: 0 };
  progTimeStats: TimeStats = { avg: 0, min: 0, max: 0 };

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

  private readonly confidenceColors: Record<string, string> = {
    '0-20%': '#ef4444',
    '20-40%': '#ff8906',
    '40-60%': '#eab308',
    '60-80%': '#3b82f6',
    '80-100%': '#2cb67d'
  };

  private readonly modelPerfColors = ['#7f5af0', '#2cb67d', '#ff8906', '#3b82f6', '#5b21b6', '#ef4444'];

  private readonly detectorColors = ['#7f5af0', '#2cb67d', '#ff8906', '#3b82f6', '#5b21b6'];
  private readonly modelColors = ['#2cb67d', '#7f5af0', '#ff8906', '#3b82f6', '#ef4444'];

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

        // Confidence distribution
        const confBuckets = data.confidence_distribution || [];
        this.totalConfCount = confBuckets.reduce((s: number, c: any) => s + c.count, 0);
        this.confidenceDist = confBuckets.map((c: any) => ({
          range: c.range || 'Unknown',
          count: c.count,
          color: this.confidenceColors[c.range] || '#9ca3af'
        }));

        // Model performance
        const perf = data.model_performance || [];
        this.modelPerf = perf.map((p: any, i: number) => ({
          model: p.model || 'Unknown',
          count: p.count,
          avg_time_ms: p.avg_time_ms || 0,
          min_time_ms: p.min_time_ms || 0,
          max_time_ms: p.max_time_ms || 0,
          avg_age_gap: p.avg_age_gap || 0,
          color: this.modelPerfColors[i % this.modelPerfColors.length]
        }));

        // Detector distribution
        const detectors = data.detector_distribution || [];
        this.totalDetectorCount = detectors.reduce((s: number, d: any) => s + d.count, 0);
        this.detectorDist = detectors.map((d: any, i: number) => ({
          detector: d.detector || 'Unknown',
          count: d.count,
          color: this.detectorColors[i % this.detectorColors.length]
        }));

        // Model distribution
        const models = data.model_distribution || [];
        this.totalModelCount = models.reduce((s: number, m: any) => s + m.count, 0);
        this.modelDist = models.map((m: any, i: number) => ({
          model: m.model || 'Unknown',
          count: m.count,
          color: this.modelColors[i % this.modelColors.length]
        }));

        // Processing time stats
        this.predTimeStats = data.prediction_time_stats || { avg: 0, min: 0, max: 0 };
        this.progTimeStats = data.progression_time_stats || { avg: 0, min: 0, max: 0 };

        checkDone();
      },
      error: () => checkDone()
    });

    // Load history for age distribution
    this.api.getHistory().subscribe({
      next: (data: any) => {
        const predictions = data.predictions || [];

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

        // Build donut segments with small gaps between them
        const segGap = this.ageDist.length > 1 ? 1 : 0;
        let offset = 0;
        this.ageDonutSegments = this.ageDist.map(d => {
          const drawPct = Math.max(d.pct - segGap, 0.5);
          const seg = { label: d.label, pct: drawPct, color: d.color, offset: offset + segGap / 2 };
          offset += d.pct;
          return seg;
        });

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

  formatTime(ms: number): string {
    if (!ms || ms === 0) return '--';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }
}

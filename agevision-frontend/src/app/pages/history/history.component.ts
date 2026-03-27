import { Component, OnInit, OnDestroy, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { TimezoneService } from '../../services/timezone.service';

interface HistoryItem {
  id: number;
  recordType: 'prediction' | 'progression';
  icon: string;
  type: string;
  age: string;
  date: Date;
  dateFormatted: string;
  model: string;
  confidence: number | null;
  imageUrl: string | null;
  processingTime: number;
  processingTimeFormatted: string;
  gender: string;
  emotion: string;
  chipClass: string;
}

@Component({
  selector: 'app-history',
  imports: [CommonModule, FormsModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class HistoryComponent implements OnInit, OnDestroy {
  loading = true;
  viewMode: 'grid' | 'list' = 'list';
  activeFilter = 'all';
  sortBy = 'newest';
  searchQuery = '';
  deleting = new Set<string>();

  pageSize = 12;
  displayCount = 12;

  items: HistoryItem[] = [];

  // Cached computed values — updated only when inputs change
  predictionCount = 0;
  progressionCount = 0;
  filteredItems: HistoryItem[] = [];
  paginatedItems: HistoryItem[] = [];
  hasMore = false;

  private destroy$ = new Subject<void>();
  private _searchDebounce: any = null;

  constructor(
    private api: ApiService,
    private tz: TimezoneService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.loadHistory();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this._searchDebounce) clearTimeout(this._searchDebounce);
  }

  loadHistory(): void {
    if (this.loading && this.items.length > 0) return;
    this.loading = true;

    this.destroy$.next();

    this.api.getHistory().pipe(takeUntil(this.destroy$)).subscribe({
      next: (data: any) => {
        const predictions = (data.predictions || []).map((p: any) => this.mapPrediction(p));
        const progressions = (data.progressions || []).map((p: any) => this.mapProgression(p));

        this.items = [...predictions, ...progressions];
        this.predictionCount = predictions.length;
        this.progressionCount = progressions.length;
        this.displayCount = this.pageSize;
        this.loading = false;
        this.recompute();
        this.cdr.markForCheck();
      },
      error: () => {
        this.loading = false;
        this.cdr.markForCheck();
      }
    });
  }

  private mapPrediction(p: any): HistoryItem {
    const date = new Date(p.created_at);
    const ms = p.processing_time_ms || 0;
    return {
      id: p.id,
      recordType: 'prediction',
      icon: p.face_count > 1 ? 'fa-solid fa-users' : 'fa-solid fa-user',
      type: 'Prediction',
      age: String(p.predicted_age),
      date,
      dateFormatted: this.tz.formatTimeAgo(date),
      model: p.detector_used || 'ensemble',
      confidence: p.confidence || 0,
      imageUrl: p.image_url || null,
      processingTime: ms,
      processingTimeFormatted: ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`,
      gender: p.gender || 'Unknown',
      emotion: p.emotion || 'Unknown',
      chipClass: 'chip-purple'
    };
  }

  private mapProgression(p: any): HistoryItem {
    const date = new Date(p.created_at);
    const ms = p.processing_time_ms || 0;
    return {
      id: p.id,
      recordType: 'progression',
      icon: 'fa-solid fa-masks-theater',
      type: 'Progression',
      age: `${p.current_age} → ${p.target_age}`,
      date,
      dateFormatted: this.tz.formatTimeAgo(date),
      model: p.model_used || 'SAM',
      confidence: null,
      imageUrl: p.original_image_url || null,
      processingTime: ms,
      processingTimeFormatted: ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`,
      gender: p.gender || 'Unknown',
      emotion: '',
      chipClass: 'chip-green'
    };
  }

  /** Recompute filtered/paginated arrays — call only when inputs change */
  private recompute(): void {
    let result = this.items;

    if (this.activeFilter !== 'all') {
      result = result.filter(i => i.recordType === this.activeFilter);
    }

    if (this.searchQuery) {
      const q = this.searchQuery.toLowerCase();
      result = result.filter(i =>
        i.type.toLowerCase().includes(q) ||
        i.model.toLowerCase().includes(q) ||
        i.age.toLowerCase().includes(q) ||
        i.gender.toLowerCase().includes(q)
      );
    }

    switch (this.sortBy) {
      case 'oldest':
        result = [...result].sort((a, b) => a.date.getTime() - b.date.getTime());
        break;
      case 'confidence':
        result = [...result].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
        break;
      default:
        result = [...result].sort((a, b) => b.date.getTime() - a.date.getTime());
    }

    this.filteredItems = result;
    this.paginatedItems = result.slice(0, this.displayCount);
    this.hasMore = this.displayCount < result.length;
  }

  setFilter(id: string): void {
    this.activeFilter = id;
    this.displayCount = this.pageSize;
    this.recompute();
  }

  onSearchInput(): void {
    if (this._searchDebounce) clearTimeout(this._searchDebounce);
    this._searchDebounce = setTimeout(() => {
      this.displayCount = this.pageSize;
      this.recompute();
      this.cdr.markForCheck();
    }, 200);
  }

  onSortChange(): void {
    this.recompute();
  }

  loadMore(): void {
    this.displayCount += this.pageSize;
    this.recompute();
  }

  toggleView(): void {
    this.viewMode = this.viewMode === 'grid' ? 'list' : 'grid';
  }

  deleteItem(item: HistoryItem): void {
    const key = `${item.recordType}-${item.id}`;
    if (this.deleting.has(key)) return;
    this.deleting.add(key);

    this.api.deleteHistory(item.id).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => {
        this.items = this.items.filter(i => !(i.id === item.id && i.recordType === item.recordType));
        this.predictionCount = this.items.filter(i => i.recordType === 'prediction').length;
        this.progressionCount = this.items.filter(i => i.recordType === 'progression').length;
        this.deleting.delete(key);
        this.recompute();
        this.cdr.markForCheck();
      },
      error: () => {
        this.deleting.delete(key);
        this.cdr.markForCheck();
      }
    });
  }

  isDeleting(item: HistoryItem): boolean {
    return this.deleting.has(`${item.recordType}-${item.id}`);
  }

  trackById(_index: number, item: HistoryItem): string {
    return `${item.recordType}-${item.id}`;
  }
}

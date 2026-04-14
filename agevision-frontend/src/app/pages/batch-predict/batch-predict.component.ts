import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';
import {
  BatchPredictionResponse,
  BatchResultItem,
  BatchSuccessItem,
} from '../../models/batch-prediction';

interface QueuedFile {
  file: File;
  preview: string;
  size: number;        // bytes
  oversize: boolean;   // local size guard
}

@Component({
  selector: 'app-batch-predict',
  imports: [CommonModule],
  templateUrl: './batch-predict.component.html',
  styleUrl: './batch-predict.component.scss',
})
export class BatchPredictComponent {
  // ─── Constants (mirror backend) ───
  readonly MAX_IMAGES = 20;
  readonly MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
  readonly MAX_FILE_SIZE_LABEL = '10 MB';

  // ─── Queue state ───
  queue: QueuedFile[] = [];
  isDragging = false;
  isProcessing = false;
  errorMessage = '';

  // ─── Results state ───
  showResults = false;
  totalImages = 0;
  totalFaces = 0;
  processingTimeMs = 0;
  batchId: string | null = null;
  results: BatchResultItem[] = [];

  constructor(private api: ApiService, private notif: NotificationService) {}

  // ─── Result narrowing helpers (used by template) ───
  isError(item: BatchResultItem): boolean {
    return !!item.error;
  }

  asSuccess(item: BatchResultItem): BatchSuccessItem {
    return item as BatchSuccessItem;
  }

  successCount(): number {
    return this.results.filter(r => !r.error).length;
  }

  errorCount(): number {
    return this.results.filter(r => !!r.error).length;
  }

  // ─── Drag & drop / file picker ───
  onDragOver(e: DragEvent): void {
    e.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(): void {
    this.isDragging = false;
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    this.isDragging = false;
    const files = Array.from(e.dataTransfer?.files || []);
    this.addFiles(files);
  }

  onFileSelect(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files) {
      this.addFiles(Array.from(input.files));
    }
    // Reset so selecting the same file again still triggers (change)
    input.value = '';
  }

  private addFiles(files: File[]): void {
    this.errorMessage = '';

    const imagesOnly = files.filter(f => f.type.startsWith('image/'));
    if (imagesOnly.length === 0) {
      this.errorMessage = 'Only image files are allowed.';
      return;
    }

    const remaining = this.MAX_IMAGES - this.queue.length;
    if (remaining <= 0) {
      this.errorMessage = `Queue is full (max ${this.MAX_IMAGES} images).`;
      return;
    }

    const toAdd = imagesOnly.slice(0, remaining);
    if (imagesOnly.length > remaining) {
      this.errorMessage = `Only added ${remaining} of ${imagesOnly.length} images — queue limit is ${this.MAX_IMAGES}.`;
    }

    toAdd.forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        this.queue = [
          ...this.queue,
          {
            file,
            preview: reader.result as string,
            size: file.size,
            oversize: file.size > this.MAX_FILE_SIZE_BYTES,
          },
        ];
      };
      reader.readAsDataURL(file);
    });
  }

  removeFromQueue(index: number): void {
    this.queue = this.queue.filter((_, i) => i !== index);
  }

  clearQueue(): void {
    this.queue = [];
    this.errorMessage = '';
  }

  // ─── Submit batch ───
  processBatch(): void {
    if (this.queue.length === 0 || this.isProcessing) return;

    this.isProcessing = true;
    this.errorMessage = '';
    this.showResults = false;

    const formData = new FormData();
    this.queue.forEach(item => formData.append('images', item.file));

    this.api.predictBatch(formData).subscribe({
      next: (res: BatchPredictionResponse) => {
        this.isProcessing = false;
        this.showResults = true;
        this.batchId = res.batch_id;
        this.totalImages = res.total_images;
        this.totalFaces = res.total_faces;
        this.processingTimeMs = res.processing_time_ms;
        this.results = res.results || [];

        const failed = this.errorCount();
        const succeeded = this.successCount();
        this.notif.success(
          'Batch Complete',
          `${succeeded} succeeded, ${failed} failed · ${this.totalFaces} face(s) detected`,
          { icon: 'fa-solid fa-layer-group', route: '/history' }
        );
      },
      error: (err) => {
        this.isProcessing = false;
        this.errorMessage =
          err?.error?.error || 'Batch processing failed. Please try again.';
        this.notif.error('Batch Failed', this.errorMessage);
      },
    });
  }

  resetAll(): void {
    this.queue = [];
    this.results = [];
    this.showResults = false;
    this.errorMessage = '';
    this.totalImages = 0;
    this.totalFaces = 0;
    this.processingTimeMs = 0;
    this.batchId = null;
  }

  // ─── Display helpers ───
  formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  formatTime(ms: number): string {
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(1)} s`;
  }

  /** Average age across ALL detected faces in successful results. */
  averageAge(): number {
    let sum = 0;
    let count = 0;
    for (const r of this.results) {
      if (r.error) continue;
      const faces = (r as BatchSuccessItem).faces || [];
      for (const f of faces) {
        sum += f.predicted_age;
        count += 1;
      }
    }
    return count > 0 ? Math.round(sum / count) : 0;
  }
}

import { Component, ElementRef, ViewChild, ChangeDetectorRef, OnDestroy, OnInit, Inject } from '@angular/core';
import { trigger, transition, style, animate } from '@angular/animations';
import { DOCUMENT } from '@angular/common';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';
import { FileTransferService } from '../../services/file-transfer.service';
import { environment } from '../../../environments/environment';
import {
  PipelineStep,
  AgingInsight,
  Progression,
  ProgressionResponse,
} from '../../models/progression';

@Component({
  selector: 'app-progress',
  imports: [CommonModule, FormsModule],
  templateUrl: './progress.component.html',
  styleUrl: './progress.component.scss',
  animations: [
    trigger('stepEnter', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-8px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ],
})
export class ProgressComponent implements OnInit, OnDestroy {
  @ViewChild('camVideo') camVideoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('camCanvas') camCanvasRef!: ElementRef<HTMLCanvasElement>;

  // ─── Source mode ───
  sourceMode: 'upload' | 'camera' = 'upload';

  // ─── Image state ───
  imagePreview: string | null = null;
  selectedFile: File | null = null;
  isDragging = false;

  // ─── Camera state ───
  cameraActive = false;
  cameraLoading = false;
  private camStream: MediaStream | null = null;

  // ─── Generation state ───
  isGenerating = false;
  showResult = false;
  errorMessage = '';
  currentStepIndex = 0;

  // ─── Target age ───
  targetAges = [5, 10, 20, 30, 40, 50, 60, 70, 80];
  selectedAge = 40;
  customAge: number | null = null;
  useCustomAge = false;

  // ─── GAN model selection ───
  gans = [
    { id: 'sam', name: 'SAM', desc: 'Style-based Age Manipulation (pSp + StyleGAN2)', icon: 'fa-solid fa-bolt' },
    { id: 'fast_aging', name: 'Fast-AgingGAN', desc: 'UTKFace trained · Indian-inclusive · Young→Old', icon: 'fa-solid fa-forward-fast' },
    { id: 'diffusion', name: 'FADING', desc: 'Diffusion-based · Bidirectional · High-quality aging', icon: 'fa-solid fa-wand-sparkles' },
  ];
  selectedGan = 'sam';

  // SAM always uses the original (FFHQ) variant
  selectedSamVariant = 'ffhq';

  // ─── Pipeline steps (streamed from backend via SSE) ───
  pipelineSteps: PipelineStep[] = [];
  elapsedSeconds = 0;
  private elapsedTimer: any = null;

  // ─── Results ───
  insights: AgingInsight[] = [];
  progressedImageUrl: string | null = null;
  currentAge: number | null = null;
  processingTime = 0;
  gender = '';
  modelUsed = '';
  progressionRecord: Progression | null = null;

  // ─── Slider comparison ───
  sliderPosition = 50;

  constructor(
    private api: ApiService,
    private cdr: ChangeDetectorRef,
    private notif: NotificationService,
    private fileTransfer: FileTransferService,
    @Inject(DOCUMENT) private document: Document,
  ) {}

  ngOnInit(): void {
    const pending = this.fileTransfer.consumePendingFile();
    if (pending) this.handleFile(pending);
  }

  ngOnDestroy(): void {
    this.stopCamera();
    this.stopElapsedTimer();
  }

  // ─── Age selection ───
  selectAge(age: number): void {
    this.selectedAge = age;
    this.useCustomAge = false;
    this.customAge = null;
  }

  get effectiveTargetAge(): number {
    if (this.useCustomAge && this.customAge && this.customAge >= 1 && this.customAge <= 100) {
      return this.customAge;
    }
    return this.selectedAge;
  }

  onCustomAgeInput(): void {
    this.useCustomAge = true;
  }

  selectGan(id: string): void {
    this.selectedGan = id;
  }



  // ─── Drag & drop ───
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
    const file = e.dataTransfer?.files[0];
    if (file) this.handleFile(file);
  }

  onFileSelect(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files?.[0]) this.handleFile(input.files[0]);
  }

  private handleFile(file: File): void {
    if (!file.type.startsWith('image/')) return;
    this.selectedFile = file;
    this.showResult = false;
    this.errorMessage = '';
    const reader = new FileReader();
    reader.onload = () => {
      this.imagePreview = reader.result as string;
    };
    reader.readAsDataURL(file);
  }

  // ─── Pipeline helpers ───
  private resetPipeline(): void {
    this.pipelineSteps = [];
    this.currentStepIndex = 0;
    this.elapsedSeconds = 0;
    if (this.elapsedTimer) clearInterval(this.elapsedTimer);
    this.elapsedTimer = setInterval(() => {
      this.elapsedSeconds++;
      this.cdr.detectChanges();
    }, 1000);
  }

  private stopElapsedTimer(): void {
    if (this.elapsedTimer) {
      clearInterval(this.elapsedTimer);
      this.elapsedTimer = null;
    }
  }

  /** Map emoji icon from backend to FontAwesome class */
  private mapIcon(icon: string): string {
    const map: Record<string, string> = {
      '\uD83E\uDDD1': 'fa-solid fa-user-clock',       // 🧑 age detection
      '\uD83D\uDC41\uFE0F': 'fa-solid fa-eye',        // 👁️ face detection
      '\uD83D\uDCCD': 'fa-solid fa-location-dot',      // 📍 face alignment
      '\uD83E\uDDEC': 'fa-solid fa-dna',               // 🧬 transform
      '\uD83D\uDCCA': 'fa-solid fa-chart-line',        // 📊 quality
    };
    return map[icon] || icon || 'fa-solid fa-gear';
  }

  /** Handle a step SSE event — insert or update the step in the list */
  private handleStepEvent(step: any): void {
    const mapped = {
      label: step.label,
      icon: this.mapIcon(step.icon),
      status: step.status as PipelineStep['status'],
      time_ms: step.time_ms,
    };

    // Find existing step by label
    const idx = this.pipelineSteps.findIndex(s => s.label === mapped.label);
    if (idx >= 0) {
      // Update existing (e.g. running → done)
      this.pipelineSteps[idx] = mapped;
    } else {
      // New step
      this.pipelineSteps.push(mapped);
    }
    this.cdr.detectChanges();
  }

  // ─── Generate progression via SSE stream ───
  async generate(): Promise<void> {
    if (!this.selectedFile) return;

    this.isGenerating = true;
    this.showResult = false;
    this.errorMessage = '';
    this.resetPipeline();

    const formData = new FormData();
    formData.append('image', this.selectedFile);
    formData.append('target_age', this.effectiveTargetAge.toString());
    formData.append('gan_model', this.selectedGan);
    if (this.selectedGan === 'sam') {
      formData.append('sam_variant', this.selectedSamVariant);
    }

    const token = localStorage.getItem('access_token');

    try {
      const response = await fetch(`${environment.apiUrl}/progress/stream/`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        throw new Error(errBody?.error || `Server error ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';  // keep incomplete part

        for (const part of parts) {
          if (!part.trim()) continue;

          let eventType = 'message';
          let data = '';

          for (const line of part.split('\n')) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              data += line.slice(6);
            }
          }

          if (!data) continue;

          try {
            const parsed = JSON.parse(data);

            switch (eventType) {
              case 'step':
                this.handleStepEvent(parsed);
                break;

              case 'age':
                this.currentAge = parsed.current_age;
                this.gender = parsed.gender;
                this.cdr.detectChanges();
                break;

              case 'result':
                this.stopElapsedTimer();
                this.isGenerating = false;
                this.showResult = true;

                const prog = parsed.progression;
                this.progressionRecord = prog;
                this.progressedImageUrl = prog.progressed_image_url;
                this.currentAge = prog.current_age;
                this.processingTime = prog.processing_time_ms;
                this.gender = prog.gender;
                this.modelUsed = prog.model_used;

                // Final pipeline steps from server
                if (parsed.steps?.length) {
                  this.pipelineSteps = parsed.steps.map((s: any) => ({
                    label: s.label,
                    icon: this.mapIcon(s.icon),
                    status: 'done' as const,
                    time_ms: s.time_ms,
                  }));
                }
                this.insights = parsed.insights || [];
                this.sliderPosition = 50;

                const timeStr = this.processingTime < 1000
                  ? `${Math.round(this.processingTime)}ms`
                  : `${(this.processingTime / 1000).toFixed(1)}s`;
                this.notif.success(
                  'Progression Complete',
                  `Aged to ${this.effectiveTargetAge} using ${this.modelUsed} in ${timeStr}`,
                  { icon: 'fa-solid fa-wand-magic-sparkles', route: '/history' }
                );
                this.cdr.detectChanges();
                break;

              case 'error':
                this.stopElapsedTimer();
                this.isGenerating = false;
                this.pipelineSteps.forEach(s => {
                  if (s.status === 'running') s.status = 'error';
                });
                this.errorMessage = parsed.error || 'Pipeline failed';
                this.notif.error('Progression Failed', this.errorMessage);
                this.cdr.detectChanges();
                break;
            }
          } catch (parseErr) {
            console.warn('SSE parse error:', parseErr, data);
          }
        }
      }

      // If stream ended without a result event
      if (this.isGenerating) {
        this.stopElapsedTimer();
        this.isGenerating = false;
        if (!this.showResult) {
          this.errorMessage = 'Connection lost before results arrived.';
          this.notif.error('Progression Failed', this.errorMessage);
        }
        this.cdr.detectChanges();
      }

    } catch (err: any) {
      this.stopElapsedTimer();
      this.isGenerating = false;
      this.pipelineSteps.forEach(s => {
        if (s.status === 'running') s.status = 'error';
      });
      this.errorMessage = err?.message || 'Something went wrong. Please try again.';
      this.notif.error('Progression Failed', this.errorMessage);
      this.cdr.detectChanges();
    }
  }

  // ─── Source mode switching ───
  switchSourceMode(m: 'upload' | 'camera'): void {
    if (m === this.sourceMode) return;
    if (this.sourceMode === 'camera') this.stopCamera();
    this.sourceMode = m;
  }

  // ─── Camera capture ───
  async startCamera(): Promise<void> {
    this.errorMessage = '';
    this.cameraLoading = true;

    try {
      this.camStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
      });

      this.cameraActive = true;
      this.cameraLoading = false;
      this.cdr.detectChanges();

      const video = this.camVideoRef.nativeElement;
      video.srcObject = this.camStream;
      await video.play();
    } catch (err: any) {
      this.cameraLoading = false;
      this.cameraActive = false;
      if (this.camStream) {
        this.camStream.getTracks().forEach(t => t.stop());
        this.camStream = null;
      }
      if (err.name === 'NotAllowedError') {
        this.errorMessage = 'Camera access denied. Please allow camera permission.';
      } else if (err.name === 'NotFoundError') {
        this.errorMessage = 'No camera found on this device.';
      } else {
        this.errorMessage = `Camera error: ${err.message}`;
      }
    }
  }

  stopCamera(): void {
    if (this.camStream) {
      this.camStream.getTracks().forEach(t => t.stop());
      this.camStream = null;
    }
    this.cameraActive = false;
  }

  capturePhoto(): void {
    if (!this.cameraActive) return;

    const video = this.camVideoRef.nativeElement;
    const canvas = this.camCanvasRef.nativeElement;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

    // Convert to File for the generate() API
    canvas.toBlob((blob) => {
      if (!blob) {
        this.errorMessage = 'Failed to capture frame from camera. Please try again.';
        return;
      }
      this.selectedFile = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
      this.imagePreview = dataUrl;
      this.showResult = false;
      this.stopCamera();
    }, 'image/jpeg', 0.92);
  }

  // ─── Clear / reset ───
  clearImage(): void {
    this.imagePreview = null;
    this.selectedFile = null;
    this.showResult = false;
    this.errorMessage = '';
    this.progressedImageUrl = null;
    this.currentAge = null;
    this.insights = [];
    this.pipelineSteps = [];
    this.stopElapsedTimer();
    this.elapsedSeconds = 0;
  }

  // ─── Download progressed image ───
  downloadImage(): void {
    if (!this.progressedImageUrl) return;
    const a = this.document.createElement('a');
    a.href = this.progressedImageUrl;
    a.download = `age_progression_${this.effectiveTargetAge}.jpg`;
    a.target = '_blank';
    this.document.body.appendChild(a);
    a.click();
    this.document.body.removeChild(a);
  }

  // ─── Download PDF report ───
  async downloadResult(): Promise<void> {
    if (!this.progressedImageUrl) return;

    const { jsPDF } = await import('jspdf');
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pw = doc.internal.pageSize.getWidth();
    const ph = doc.internal.pageSize.getHeight();
    const margin = 16;
    const contentW = pw - margin * 2;
    let y = 0;

    // ── Theme palette (matches dark UI) ──
    const bg       = [6, 6, 15];
    const cardFill = [16, 14, 36];
    const cardBdr  = [38, 36, 58];
    const accent1  = [127, 90, 240];
    const accent2  = [44, 182, 125];
    const white    = [255, 255, 255];
    const textSec  = [224, 223, 230];
    const muted    = [155, 155, 175];
    const divider  = [38, 36, 58];

    // ── Helpers ──
    const loadImage = (url: string): Promise<string> =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          const c = this.document.createElement('canvas');
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
          c.getContext('2d')!.drawImage(img, 0, 0);
          resolve(c.toDataURL('image/jpeg', 0.92));
        };
        img.onerror = reject;
        img.src = url;
      });

    const fillRect = (x: number, ry: number, w: number, h: number, r: number, fill: number[]) => {
      doc.setFillColor(fill[0], fill[1], fill[2]);
      doc.roundedRect(x, ry, w, h, r, r, 'F');
    };

    const drawPageBg = () => {
      doc.setFillColor(bg[0], bg[1], bg[2]);
      doc.rect(0, 0, pw, ph, 'F');
      // Subtle corner orbs
      doc.setGState(doc.GState({ opacity: 0.06 }));
      doc.setFillColor(accent1[0], accent1[1], accent1[2]);
      doc.circle(-10, -10, 55, 'F');
      doc.setFillColor(accent2[0], accent2[1], accent2[2]);
      doc.circle(pw + 10, ph + 10, 45, 'F');
      doc.setGState(doc.GState({ opacity: 1 }));
    };

    const glassCard = (x: number, ry: number, w: number, h: number) => {
      fillRect(x, ry, w, h, 5, cardFill);
      doc.setGState(doc.GState({ opacity: 0.06 }));
      doc.setFillColor(255, 255, 255);
      doc.roundedRect(x, ry, w, 1, 5, 5, 'F');
      doc.setGState(doc.GState({ opacity: 1 }));
      doc.setDrawColor(cardBdr[0], cardBdr[1], cardBdr[2]);
      doc.setLineWidth(0.3);
      doc.roundedRect(x, ry, w, h, 5, 5, 'S');
    };

    // Section title with small accent dot instead of unicode icons
    const sectionTitle = (title: string, gradientWord: string) => {
      // Accent dot
      doc.setFillColor(accent1[0], accent1[1], accent1[2]);
      doc.circle(margin + 1.5, y - 1.2, 1.5, 'F');

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      const base = title.replace(gradientWord, '');
      doc.setTextColor(white[0], white[1], white[2]);
      doc.text(base, margin + 6, y);
      const baseW = doc.getTextWidth(base);
      doc.setTextColor(accent1[0], accent1[1], accent1[2]);
      doc.text(gradientWord, margin + 6 + baseW, y);
      y += 5;
    };

    const pill = (px: number, py: number, text: string, bgc: number[], fgc: number[]) => {
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(6.5);
      const tw = doc.getTextWidth(text) + 5;
      fillRect(px, py - 2.8, tw, 5, 2.5, bgc);
      doc.setTextColor(fgc[0], fgc[1], fgc[2]);
      doc.text(text, px + 2.5, py + 0.3);
      return tw;
    };

    // Draw a right-pointing triangle arrow (no unicode needed)
    const drawArrow = (cx: number, cy: number) => {
      fillRect(cx - 4.5, cy - 4.5, 9, 9, 4.5, accent1);
      doc.setFillColor(255, 255, 255);
      doc.triangle(cx - 2, cy - 2.5, cx - 2, cy + 2.5, cx + 3, cy, 'F');
    };

    // Check if content fits, add new page if needed
    const ensureSpace = (needed: number) => {
      const footerReserve = 30; // space for footer
      if (y + needed > ph - footerReserve) {
        drawFooter();
        doc.addPage();
        drawPageBg();
        y = margin;
      }
    };

    // Footer renderer (called per page)
    const drawFooter = () => {
      const footerH = 16;
      const fy = ph - footerH - 8;

      glassCard(margin, fy, contentW, footerH);
      fillRect(margin, fy, 2.5, footerH, 0, accent1);

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7);
      doc.setTextColor(white[0], white[1], white[2]);
      doc.text('AgeVision AI', margin + 7, fy + 6.5);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(6);
      doc.setTextColor(muted[0], muted[1], muted[2]);
      doc.text('Age Invariant Face Recognition & Retrieval System', margin + 7, fy + 11);

      // Right side: confidential + date
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(6);
      const confText = 'CONFIDENTIAL';
      const confW = doc.getTextWidth(confText) + 5;
      fillRect(pw - margin - 6 - confW, fy + 3, confW, 5, 2.5, [50, 20, 20]);
      doc.setTextColor(239, 100, 100);
      doc.text(confText, pw - margin - 6 - confW + 2.5, fy + 6.5);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(5.5);
      doc.setTextColor(muted[0], muted[1], muted[2]);
      doc.text(new Date().toISOString().split('T')[0], pw - margin - 6, fy + 12, { align: 'right' });
    };

    // ═══════════════════════════════════════
    //  PAGE 1 BACKGROUND
    // ═══════════════════════════════════════
    drawPageBg();

    // ═══════════════════════════════════════
    //  HEADER
    // ═══════════════════════════════════════
    const headerH = 24;
    glassCard(margin, 8, contentW, headerH);
    fillRect(margin, 8, 2.5, headerH, 0, accent1);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(white[0], white[1], white[2]);
    doc.text('AgeVision', margin + 7, 17);
    const avW = doc.getTextWidth('AgeVision');
    doc.setTextColor(accent1[0], accent1[1], accent1[2]);
    doc.text(' AI', margin + 7 + avW, 17);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(muted[0], muted[1], muted[2]);
    doc.text('Age Progression Report', margin + 7, 23);

    // Date
    const dateStr = new Date().toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
    doc.setFontSize(6.5);
    doc.text(dateStr, pw - margin - 6, 17, { align: 'right' });

    // Report ID
    const reportId = `#RPT-${Date.now().toString(36).toUpperCase().slice(-6)}`;
    doc.setFontSize(6.5);
    const ridW = doc.getTextWidth(reportId) + 5;
    pill(pw - margin - 6 - ridW, 23, reportId, accent1, white);

    y = 8 + headerH + 7;

    // ═══════════════════════════════════════
    //  BEFORE & AFTER COMPARISON
    // ═══════════════════════════════════════
    sectionTitle('Before & After ', 'Comparison');

    const imgCardH = 68;
    const halfW = (contentW - 6) / 2;
    glassCard(margin, y, contentW, imgCardH);

    let originalData: string | null = null;
    let progressedData: string | null = null;
    try {
      if (this.imagePreview) originalData = this.imagePreview.startsWith('data:')
        ? this.imagePreview : await loadImage(this.imagePreview);
      if (this.progressedImageUrl) progressedData = await loadImage(this.progressedImageUrl);
    } catch { /* skip */ }

    const imgPad = 4;
    const imgH = imgCardH - 16;
    const imgW = halfW - imgPad * 2;

    fillRect(margin + imgPad, y + imgPad, imgW, imgH, 3, bg);
    if (originalData) {
      doc.addImage(originalData, 'JPEG', margin + imgPad, y + imgPad, imgW, imgH, undefined, 'MEDIUM');
    }

    fillRect(margin + halfW + 6 + imgPad, y + imgPad, imgW, imgH, 3, bg);
    if (progressedData) {
      doc.addImage(progressedData, 'JPEG', margin + halfW + 6 + imgPad, y + imgPad, imgW, imgH, undefined, 'MEDIUM');
    }

    // Labels
    const lblY = y + imgCardH - 4;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(7);
    doc.setTextColor(textSec[0], textSec[1], textSec[2]);
    doc.text(`Original (Age ${this.currentAge || '?'})`, margin + imgPad + imgW / 2, lblY, { align: 'center' });
    doc.setTextColor(accent2[0], accent2[1], accent2[2]);
    doc.text(`Aged to ${this.effectiveTargetAge}`, margin + halfW + 6 + imgPad + imgW / 2, lblY, { align: 'center' });

    // Arrow between images (triangle, no unicode)
    drawArrow(margin + halfW + 3, y + imgCardH / 2 - 2);

    y += imgCardH + 7;

    // ═══════════════════════════════════════
    //  PROCESSING DETAILS
    // ═══════════════════════════════════════
    sectionTitle('Processing ', 'Details');

    const stats = [
      { label: 'Model Used',      value: this.modelUsed || 'N/A',      accent: false },
      { label: 'Detected Age',    value: `${this.currentAge ?? 'N/A'}`, accent: true  },
      { label: 'Target Age',      value: `${this.effectiveTargetAge}`,  accent: true  },
      { label: 'Gender',          value: this.gender || 'N/A',          accent: false },
      { label: 'Processing Time', value: `${Math.round(this.processingTime)} ms`, accent: false },
      { label: 'GAN Model',       value: this.selectedGan === 'sam' ? 'SAM' : this.selectedGan === 'diffusion' ? 'FADING' : 'Fast-AgingGAN', accent: false },
    ];

    const rowH = 7;
    const statsCardH = stats.length * rowH + 7;
    ensureSpace(statsCardH + 12);
    glassCard(margin, y, contentW, statsCardH);

    stats.forEach((s, i) => {
      const ry = y + 5.5 + i * rowH;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.5);
      doc.setTextColor(muted[0], muted[1], muted[2]);
      doc.text(s.label, margin + 8, ry);

      doc.setFont('helvetica', 'bold');
      doc.setTextColor(s.accent ? accent1[0] : white[0], s.accent ? accent1[1] : white[1], s.accent ? accent1[2] : white[2]);
      doc.text(s.value, pw - margin - 8, ry, { align: 'right' });

      if (i < stats.length - 1) {
        doc.setDrawColor(divider[0], divider[1], divider[2]);
        doc.setLineWidth(0.15);
        doc.line(margin + 8, ry + 3.2, pw - margin - 8, ry + 3.2);
      }
    });

    y += statsCardH + 7;

    // ═══════════════════════════════════════
    //  AGING INSIGHTS
    // ═══════════════════════════════════════
    if (this.insights.length) {
      const insRowH = 9;
      const insCardH = this.insights.length * insRowH + 7;
      ensureSpace(insCardH + 12);
      sectionTitle('Aging ', 'Insights');
      glassCard(margin, y, contentW, insCardH);

      this.insights.forEach((ins, i) => {
        const ry = y + 6 + i * insRowH;

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.5);
        doc.setTextColor(white[0], white[1], white[2]);
        doc.text(ins.label, margin + 8, ry);

        doc.setFont('helvetica', 'bold');
        doc.text(`${ins.value}%`, pw - margin - 8, ry, { align: 'right' });

        // Bar track
        const barY = ry + 2;
        const barW = contentW - 16;
        fillRect(margin + 8, barY, barW, 2.2, 1.1, [25, 24, 45]);

        // Bar fill
        const hex = ins.color || '#7f5af0';
        const cr = parseInt(hex.slice(1, 3), 16) || accent1[0];
        const cg = parseInt(hex.slice(3, 5), 16) || accent1[1];
        const cb = parseInt(hex.slice(5, 7), 16) || accent1[2];
        const fillW = Math.max(1, (ins.value / 100) * barW);
        fillRect(margin + 8, barY, fillW, 2.2, 1.1, [cr, cg, cb]);
      });

      y += insCardH + 7;
    }

    // ═══════════════════════════════════════
    //  FOOTER (drawn on current/last page)
    // ═══════════════════════════════════════
    drawFooter();

    // ── Save ──
    doc.save(`AgeVision_Progression_Report_${this.effectiveTargetAge}.pdf`);
  }

  // ─── Slider comparison ───
  onSliderMove(e: MouseEvent, container: HTMLElement): void {
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    this.sliderPosition = Math.max(0, Math.min(100, (x / rect.width) * 100));
  }
}

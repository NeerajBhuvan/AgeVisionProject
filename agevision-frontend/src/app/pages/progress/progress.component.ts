import { Component, ElementRef, ViewChild, ChangeDetectorRef, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';
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
})
export class ProgressComponent implements OnDestroy {
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
  ];
  selectedGan = 'sam';

  // ─── Pipeline steps (animated during generation) ───
  pipelineSteps: PipelineStep[] = [
    { label: 'Face Alignment (FFHQ/dlib)', icon: 'fa-solid fa-location-dot', status: 'pending' },
    { label: 'SAM Age Transform', icon: 'fa-solid fa-dna', status: 'pending' },
    { label: 'Quality Assessment', icon: 'fa-solid fa-chart-line', status: 'pending' },
  ];

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

  constructor(private api: ApiService, private cdr: ChangeDetectorRef, private notif: NotificationService) {}

  ngOnDestroy(): void {
    this.stopCamera();
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

  // ─── Pipeline animation ───
  private resetPipeline(): void {
    this.pipelineSteps = this.pipelineSteps.map((s) => ({
      ...s,
      status: 'pending' as const,
      time_ms: undefined,
    }));
    this.currentStepIndex = 0;
  }

  private animatePipeline(): void {
    const interval = setInterval(() => {
      if (this.currentStepIndex >= this.pipelineSteps.length) {
        clearInterval(interval);
        return;
      }
      // Mark previous step as done
      if (this.currentStepIndex > 0) {
        this.pipelineSteps[this.currentStepIndex - 1].status = 'done';
      }
      // Mark current step as running
      this.pipelineSteps[this.currentStepIndex].status = 'running';
      this.currentStepIndex++;
    }, 600);
  }

  private finalizePipeline(serverSteps: PipelineStep[]): void {
    // Replace with actual server steps if available
    if (serverSteps && serverSteps.length > 0) {
      this.pipelineSteps = serverSteps.map((s) => ({
        label: s.label,
        icon: s.icon || 'fa-solid fa-check',
        status: s.status || 'done',
        time_ms: s.time_ms,
      }));
    }
    // Ensure all are marked done
    this.pipelineSteps.forEach((s) => (s.status = 'done'));
  }

  // ─── Generate progression ───
  generate(): void {
    if (!this.selectedFile) return;

    this.isGenerating = true;
    this.showResult = false;
    this.errorMessage = '';
    this.resetPipeline();
    this.animatePipeline();

    const formData = new FormData();
    formData.append('image', this.selectedFile);
    formData.append('target_age', this.effectiveTargetAge.toString());
    formData.append('gan_model', this.selectedGan);

    this.api.progressAge(formData).subscribe({
      next: (res: ProgressionResponse) => {
        this.isGenerating = false;
        this.showResult = true;

        const prog = res.progression;
        this.progressionRecord = prog;
        this.progressedImageUrl = prog.progressed_image_url;
        this.currentAge = prog.current_age;
        this.processingTime = prog.processing_time_ms;
        this.gender = prog.gender;
        this.modelUsed = prog.model_used;

        this.finalizePipeline(res.steps || []);
        this.insights = res.insights || [];
        this.sliderPosition = 50;

        const timeStr = this.processingTime < 1000
          ? `${Math.round(this.processingTime)}ms`
          : `${(this.processingTime / 1000).toFixed(1)}s`;
        this.notif.success(
          'Progression Complete',
          `Aged to ${this.effectiveTargetAge} using ${this.modelUsed} in ${timeStr}`,
          { icon: 'fa-solid fa-wand-magic-sparkles', route: '/history' }
        );
      },
      error: (err) => {
        this.isGenerating = false;
        this.pipelineSteps.forEach((s) => {
          if (s.status === 'running') s.status = 'error';
        });
        this.errorMessage =
          err?.error?.error || 'Something went wrong. Please try again.';
        this.notif.error('Progression Failed', this.errorMessage);
      },
    });
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
      if (!blob) return;
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
    this.resetPipeline();
  }

  // ─── Download progressed image ───
  downloadResult(): void {
    if (!this.progressedImageUrl) return;
    const a = document.createElement('a');
    a.href = this.progressedImageUrl;
    a.download = `age_progression_${this.effectiveTargetAge}.jpg`;
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // ─── Slider comparison ───
  onSliderMove(e: MouseEvent, container: HTMLElement): void {
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    this.sliderPosition = Math.max(0, Math.min(100, (x / rect.width) * 100));
  }
}

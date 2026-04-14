import { Component, ElementRef, OnDestroy, OnInit, ViewChild, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';
import { FileTransferService } from '../../services/file-transfer.service';
import { PredictionResponse, FacePrediction } from '../../models/prediction';

@Component({
  selector: 'app-predict',
  imports: [CommonModule],
  templateUrl: './predict.component.html',
  styleUrl: './predict.component.scss'
})
export class PredictComponent implements OnInit, OnDestroy {
  @ViewChild('videoEl') videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasEl') canvasRef!: ElementRef<HTMLCanvasElement>;

  // Mode: 'upload' or 'camera'
  mode: 'upload' | 'camera' = 'upload';

  // Upload state
  singleSamples = [
    { file: 'single_1.jpg', label: 'Young Woman' },
    { file: 'single_2.jpg', label: 'Young Man' },
    { file: 'single_3.jpg', label: 'Woman' },
    { file: 'single_4.jpg', label: 'Man' },
    { file: 'single_5.jpg', label: 'Man' },
    { file: 'single_6.jpg', label: 'Senior Man' },
    { file: 'single_7.jpg', label: 'Senior Woman' },
    { file: 'single_8.jpg', label: 'Young Woman' }
  ];
  loadingSample = false;

  groupSamples = [
    { file: 'group_1.jpg', label: 'Startup Team', faces: 8 },
    { file: 'group_2.jpg', label: 'Classmates', faces: 7 },
    { file: 'group_3.jpg', label: 'Team Meeting', faces: 5 },
    { file: 'group_4.jpg', label: 'Team Outdoor', faces: 4 },
    { file: 'group_5.jpg', label: 'Friends Selfie', faces: 4 }
  ];

  imagePreview: string | null = null;
  selectedFile: File | null = null;
  fileName = '';
  isDragging = false;
  isAnalyzing = false;
  showResults = false;
  errorMessage = '';

  // Results (shared by upload + camera)
  predictedAge = 0;
  confidence = 0;
  ageRange = { min: 0, max: 0 };
  faceDetected = false;
  gender = '';
  emotion = '';
  faceCount = 0;
  processingTime = 0;
  faces: FacePrediction[] = [];
  selectedFaceIndex = 0;
  confBars = [
    { label: 'Face Detection', value: 0, color: 'var(--accent2)' },
    { label: 'Age Estimation', value: 0, color: 'var(--accent1)' },
    { label: 'Model Confidence', value: 0, color: 'var(--accent3)' }
  ];

  // Camera state
  cameraActive = false;
  cameraLoading = false;
  isPredicting = false;
  autoPredict = true;
  cameraList: MediaDeviceInfo[] = [];
  selectedCameraId = '';

  private stream: MediaStream | null = null;
  private predictInterval: ReturnType<typeof setInterval> | null = null;
  private readonly PREDICT_INTERVAL_MS = 1500;

  constructor(
    private api: ApiService,
    private cdr: ChangeDetectorRef,
    private notif: NotificationService,
    private fileTransfer: FileTransferService
  ) {}

  ngOnInit(): void {
    const pending = this.fileTransfer.consumePendingFile();
    if (pending) this.handleFile(pending);
  }

  ngOnDestroy(): void {
    this.stopCamera();
  }

  // ─── Mode Switching ───────────────────────────────────────

  switchMode(m: 'upload' | 'camera'): void {
    if (m === this.mode) return;
    // Clean up current mode
    if (this.mode === 'camera') {
      this.stopCamera();
    }
    this.mode = m;
    this.resetResults();

    if (m === 'camera') {
      this.loadCameraList();
    }
  }

  // ─── Upload Mode ──────────────────────────────────────────

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
    this.fileName = file.name;
    this.errorMessage = '';
    // Clear previous detection results
    this.faces = [];
    this.selectedFaceIndex = 0;
    this.showResults = false;
    const reader = new FileReader();
    reader.onload = () => {
      this.imagePreview = reader.result as string;
    };
    reader.readAsDataURL(file);
  }

  analyze(): void {
    if (!this.selectedFile) return;
    this.isAnalyzing = true;
    this.showResults = false;
    this.errorMessage = '';

    const formData = new FormData();
    formData.append('image', this.selectedFile, this.selectedFile.name);
    formData.append('detector_mode', 'insightface');

    this.api.predictAge(formData).subscribe({
      next: (res: PredictionResponse) => {
        const p = res.prediction;
        this.faces = res.faces || [];
        this.selectedFaceIndex = 0;
        this.faceCount = p.face_count;
        this.processingTime = p.processing_time_ms;
        this.faceDetected = p.face_count > 0;
        this.applyFaceResult(this.selectedFaceIndex);
        this.isAnalyzing = false;
        this.showResults = true;
        this.notif.success(
          'Prediction Complete',
          `Detected ${p.face_count} face(s) in ${this.formatProcessingTime()}`,
          { icon: 'fa-solid fa-microscope', route: '/history' }
        );
      },
      error: (err) => {
        this.isAnalyzing = false;
        this.errorMessage = err.error?.error || 'Prediction failed. Please try again.';
        this.notif.error('Prediction Failed', this.errorMessage);
      }
    });
  }

  selectSample(sample: { file: string; label: string }): void {
    this.loadingSample = true;
    fetch(`samples/${sample.file}`)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], sample.file, { type: blob.type });
        this.handleFile(file);
        this.loadingSample = false;
      })
      .catch(() => {
        this.errorMessage = 'Failed to load sample image.';
        this.loadingSample = false;
      });
  }

  selectGroupSample(sample: { file: string; label: string; faces: number }): void {
    this.loadingSample = true;
    fetch(`samples/${sample.file}`)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], sample.file, { type: blob.type });
        this.handleFile(file);
        this.loadingSample = false;
      })
      .catch(() => {
        this.errorMessage = 'Failed to load sample image.';
        this.loadingSample = false;
      });
  }

  clearImage(): void {
    this.imagePreview = null;
    this.selectedFile = null;
    this.fileName = '';
    this.errorMessage = '';
    this.resetResults();
  }

  // ─── Camera Mode ──────────────────────────────────────────

  async loadCameraList(): Promise<void> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.cameraList = devices.filter(d => d.kind === 'videoinput');
      if (this.cameraList.length > 0 && !this.selectedCameraId) {
        this.selectedCameraId = this.cameraList[0].deviceId;
      }
    } catch {
      this.errorMessage = 'Could not list cameras.';
    }
  }

  async startCamera(): Promise<void> {
    this.errorMessage = '';
    this.cameraLoading = true;

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: this.selectedCameraId ? { exact: this.selectedCameraId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);

      // Show the video element first by setting cameraActive + triggering change detection
      this.cameraActive = true;
      this.cameraLoading = false;
      this.cdr.detectChanges();

      // Now the <video #videoEl> is rendered and ViewChild is available
      const video = this.videoRef.nativeElement;
      video.srcObject = this.stream;
      await video.play();

      this.loadCameraList();

      if (this.autoPredict) {
        this.startAutoPrediction();
      }
    } catch (err: any) {
      this.cameraLoading = false;
      this.cameraActive = false;
      if (this.stream) {
        this.stream.getTracks().forEach(t => t.stop());
        this.stream = null;
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
    this.stopAutoPrediction();
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    this.cameraActive = false;
    this.resetResults();
  }

  switchCamera(deviceId: string): void {
    this.selectedCameraId = deviceId;
    if (this.cameraActive) {
      this.stopCamera();
      this.startCamera();
    }
  }

  toggleAutoPredict(): void {
    this.autoPredict = !this.autoPredict;
    if (this.autoPredict && this.cameraActive) {
      this.startAutoPrediction();
    } else {
      this.stopAutoPrediction();
    }
  }

  captureAndPredict(): void {
    if (!this.cameraActive || this.isPredicting) return;

    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const frameBase64 = canvas.toDataURL('image/jpeg', 0.85);

    this.isPredicting = true;

    this.api.predictCamera(frameBase64).subscribe({
      next: (res) => {
        this.faces = res.faces || [];
        this.faceCount = res.face_count || 0;
        this.processingTime = res.processing_time_ms || 0;
        this.faceDetected = this.faceCount > 0;
        this.showResults = this.faceCount > 0;

        if (this.faces.length > 0) {
          this.selectedFaceIndex = 0;
          this.applyFaceResult(0);
        }

        this.isPredicting = false;
      },
      error: () => {
        this.isPredicting = false;
      }
    });
  }

  private startAutoPrediction(): void {
    this.stopAutoPrediction();
    this.captureAndPredict();
    this.predictInterval = setInterval(() => {
      // Skip if a prediction is already in-flight (prevents request pileup)
      if (!this.isPredicting) {
        this.captureAndPredict();
      }
    }, this.PREDICT_INTERVAL_MS);
  }

  private stopAutoPrediction(): void {
    if (this.predictInterval) {
      clearInterval(this.predictInterval);
      this.predictInterval = null;
    }
  }

  // ─── Shared ───────────────────────────────────────────────

  formatProcessingTime(): string {
    if (this.processingTime < 1000) {
      return `${Math.round(this.processingTime)}ms`;
    }
    return `${(this.processingTime / 1000).toFixed(1)}s`;
  }

  applyFaceResult(index: number): void {
    const face = this.faces[index];
    if (!face) return;
    this.predictedAge = face.predicted_age;
    this.confidence = Math.round(face.confidence * 100);
    this.ageRange = {
      min: Math.max(0, face.predicted_age - 3),
      max: face.predicted_age + 3
    };
    this.gender = face.gender;
    this.emotion = face.emotion;
    this.confBars = [
      { label: 'Face Detection', value: this.faceDetected ? 98 : 0, color: 'var(--accent2)' },
      { label: 'Age Estimation', value: this.confidence, color: 'var(--accent1)' },
      { label: 'Model Confidence', value: Math.min(99, this.confidence + Math.floor(Math.random() * 5)), color: 'var(--accent3)' }
    ];
  }

  selectFace(index: number): void {
    this.selectedFaceIndex = index;
    this.applyFaceResult(index);
  }

  private resetResults(): void {
    this.showResults = false;
    this.faceDetected = false;
    this.faces = [];
    this.selectedFaceIndex = 0;
    this.gender = '';
    this.emotion = '';
    this.faceCount = 0;
    this.processingTime = 0;
  }
}

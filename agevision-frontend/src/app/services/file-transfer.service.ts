import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class FileTransferService {
  private pendingFile: File | null = null;

  setPendingFile(file: File): void {
    this.pendingFile = file;
  }

  consumePendingFile(): File | null {
    const file = this.pendingFile;
    this.pendingFile = null;
    return file;
  }
}

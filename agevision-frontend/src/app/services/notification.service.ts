import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { AppNotification, NotificationType } from '../models/notification';

@Injectable({ providedIn: 'root' })
export class NotificationService {
  private readonly STORAGE_KEY = 'agevision_notifications';
  private readonly MAX_NOTIFICATIONS = 50;

  private notificationsSubject = new BehaviorSubject<AppNotification[]>([]);
  public notifications$ = this.notificationsSubject.asObservable();

  private toastSubject = new BehaviorSubject<AppNotification | null>(null);
  public toast$ = this.toastSubject.asObservable();

  constructor() {
    this.loadFromStorage();
  }

  get unreadCount(): number {
    return this.notificationsSubject.value.filter(n => !n.read).length;
  }

  get notifications(): AppNotification[] {
    return this.notificationsSubject.value;
  }

  /** Add a notification and show a toast */
  push(type: NotificationType, title: string, message: string, options?: { icon?: string; route?: string }): void {
    const notification: AppNotification = {
      id: this.generateId(),
      type,
      title,
      message,
      icon: options?.icon || this.getDefaultIcon(type),
      timestamp: new Date(),
      read: false,
      route: options?.route
    };

    const list = [notification, ...this.notificationsSubject.value].slice(0, this.MAX_NOTIFICATIONS);
    this.notificationsSubject.next(list);
    this.saveToStorage(list);

    // Show toast
    this.toastSubject.next(notification);
  }

  /** Shorthand methods */
  success(title: string, message: string, options?: { icon?: string; route?: string }): void {
    this.push('success', title, message, options);
  }

  error(title: string, message: string, options?: { icon?: string; route?: string }): void {
    this.push('error', title, message, options);
  }

  info(title: string, message: string, options?: { icon?: string; route?: string }): void {
    this.push('info', title, message, options);
  }

  warning(title: string, message: string, options?: { icon?: string; route?: string }): void {
    this.push('warning', title, message, options);
  }

  /** Mark a single notification as read */
  markAsRead(id: string): void {
    const list = this.notificationsSubject.value.map(n =>
      n.id === id ? { ...n, read: true } : n
    );
    this.notificationsSubject.next(list);
    this.saveToStorage(list);
  }

  /** Mark all notifications as read */
  markAllAsRead(): void {
    const list = this.notificationsSubject.value.map(n => ({ ...n, read: true }));
    this.notificationsSubject.next(list);
    this.saveToStorage(list);
  }

  /** Remove a single notification */
  remove(id: string): void {
    const list = this.notificationsSubject.value.filter(n => n.id !== id);
    this.notificationsSubject.next(list);
    this.saveToStorage(list);
  }

  /** Clear all notifications */
  clearAll(): void {
    this.notificationsSubject.next([]);
    this.saveToStorage([]);
  }

  /** Dismiss the current toast */
  dismissToast(): void {
    this.toastSubject.next(null);
  }

  /** Format relative time (e.g. "2m ago") */
  timeAgo(date: Date): string {
    const now = new Date();
    const diff = Math.floor((now.getTime() - new Date(date).getTime()) / 1000);
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  private getDefaultIcon(type: NotificationType): string {
    switch (type) {
      case 'success': return 'fa-solid fa-circle-check';
      case 'error': return 'fa-solid fa-circle-xmark';
      case 'warning': return 'fa-solid fa-triangle-exclamation';
      case 'info': return 'fa-solid fa-circle-info';
    }
  }

  private generateId(): string {
    return `notif_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  }

  private saveToStorage(list: AppNotification[]): void {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(list));
    } catch { /* storage full — ignore */ }
  }

  private loadFromStorage(): void {
    try {
      const raw = localStorage.getItem(this.STORAGE_KEY);
      if (raw) {
        const parsed: AppNotification[] = JSON.parse(raw);
        this.notificationsSubject.next(parsed);
      }
    } catch { /* corrupted — ignore */ }
  }
}

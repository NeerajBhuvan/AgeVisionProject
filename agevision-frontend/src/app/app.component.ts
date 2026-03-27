import { Component, OnInit, OnDestroy } from '@angular/core';
import { RouterOutlet, Router, NavigationEnd, RouterLink, RouterLinkActive, ChildrenOutletContexts } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subscription, filter } from 'rxjs';
import { SidebarComponent } from './shared/sidebar/sidebar.component';
import { TopbarComponent } from './shared/topbar/topbar.component';
import { ThemeService } from './services/theme.service';
import { routeAnimations } from './route-animations';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, CommonModule, SidebarComponent, TopbarComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
  animations: [routeAnimations]
})
export class AppComponent implements OnInit, OnDestroy {
  isAuthPage = false;
  sidebarOpen = false;
  private routeSub!: Subscription;

  constructor(private router: Router, private themeService: ThemeService, private contexts: ChildrenOutletContexts) {}

  getRouteAnimationData() {
    return this.contexts.getContext('primary')?.route?.snapshot?.data?.['animation'];
  }

  ngOnInit(): void {
    this.routeSub = this.router.events
      .pipe(filter(e => e instanceof NavigationEnd))
      .subscribe((e) => {
        const url = (e as NavigationEnd).urlAfterRedirects || (e as NavigationEnd).url;
        this.isAuthPage = url.includes('/login') || url.includes('/register') || url.includes('/forgot-password');
        if (this.isAuthPage) this.sidebarOpen = false;
      });
  }

  toggleSidebar(): void {
    this.sidebarOpen = !this.sidebarOpen;
  }

  closeSidebar(): void {
    this.sidebarOpen = false;
  }

  ngOnDestroy(): void {
    this.routeSub?.unsubscribe();
  }
}

import {
  trigger,
  transition,
  style,
  animate,
  query,
  group
} from '@angular/animations';

export const routeAnimations = trigger('routeAnimation', [
  transition('* <=> *', [
    query(':enter, :leave', [
      style({
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%'
      })
    ], { optional: true }),

    group([
      query(':leave', [
        style({ opacity: 1, transform: 'scale(1)' }),
        animate('180ms ease-in',
          style({ opacity: 0, transform: 'scale(0.98)' })
        )
      ], { optional: true }),

      query(':enter', [
        style({ opacity: 0, transform: 'translateY(12px)' }),
        animate('280ms 80ms cubic-bezier(0.35, 0, 0.25, 1)',
          style({ opacity: 1, transform: 'translateY(0)' })
        )
      ], { optional: true })
    ])
  ])
]);

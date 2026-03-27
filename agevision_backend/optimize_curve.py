"""
Find optimal correction approach by simulating different curves and parameters.
"""
import json
import numpy as np

data = json.load(open('media/test_images/accuracy_report.json'))

labeled = []
for r in data['results']:
    gt = r.get('ground_truth_age')
    if not gt:
        continue
    ages = [float(a) for a in r['ensemble_ages']]
    labeled.append({
        'name': r['description'][:25],
        'gt': gt,
        'ages': ages,
    })


def compute_weighted(ages, mean_w, max_w, med_w):
    a = np.array(ages)
    return mean_w * np.mean(a) + max_w * np.max(a) + med_w * np.median(a)


def build_curve(points, bin_width=6):
    pts = np.array(points)
    pts = pts[pts[:, 0].argsort()]
    bins_raw, bins_true = [], []
    lo = pts[0, 0]
    while lo <= pts[-1, 0]:
        hi = lo + bin_width
        mask = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if mask.any():
            bins_raw.append(float(np.median(pts[mask, 0])))
            bins_true.append(float(np.median(pts[mask, 1])))
        lo = hi
    for i in range(1, len(bins_true)):
        if bins_true[i] < bins_true[i - 1]:
            bins_true[i] = bins_true[i - 1] + 1
    raw_arr = [15.0] + bins_raw + [70.0]
    true_arr = [15.0] + bins_true + [max(bins_true[-1] + 15, 85.0)]
    return np.array(raw_arr), np.array(true_arr)


def simulate(mean_w, max_w, med_w, cal_points, bin_width=6):
    corr_raw, corr_true = build_curve(cal_points, bin_width)
    errors = []
    for s in labeled:
        weighted = compute_weighted(s['ages'], mean_w, max_w, med_w)
        corrected = float(np.interp(weighted, corr_raw, corr_true))
        pred = max(1, min(100, int(round(corrected))))
        errors.append(abs(pred - s['gt']))
    mae = np.mean(errors)
    within5 = 100.0 * sum(1 for e in errors if e <= 5) / len(errors)
    within10 = 100.0 * sum(1 for e in errors if e <= 10) / len(errors)
    return mae, within5, within10, errors


# Try different weight combos and bin widths
print("=== Searching for optimal parameters ===\n")
best_mae = 999
best_config = None
results = []

for mean_w in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for max_w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        med_w = 1.0 - mean_w - max_w
        if med_w < 0:
            continue
        # Build cal points for this weight combo
        cal = []
        for s in labeled:
            w = compute_weighted(s['ages'], mean_w, max_w, med_w)
            cal.append([w, s['gt']])

        for bw in [5, 6, 7, 8]:
            mae, w5, w10, errs = simulate(mean_w, max_w, med_w, cal, bw)
            results.append((mae, w5, w10, mean_w, max_w, med_w, bw, errs))
            if mae < best_mae:
                best_mae = mae
                best_config = (mean_w, max_w, med_w, bw)

results.sort(key=lambda x: x[0])

print("Top 10 configurations:")
for i, (mae, w5, w10, mw, xw, dw, bw, _) in enumerate(results[:10]):
    print(f"  {i+1}. MAE={mae:.2f} ±5yr={w5:.0f}% ±10yr={w10:.0f}%  "
          f"mean={mw:.2f} max={xw:.2f} med={dw:.2f} bin={bw}")

print(f"\nBest: mean_w={best_config[0]}, max_w={best_config[1]}, "
      f"med_w={best_config[2]}, bin_width={best_config[3]}")

# Show per-sample errors for best config
mw, xw, dw, bw = best_config
cal = [[compute_weighted(s['ages'], mw, xw, dw), s['gt']] for s in labeled]
mae, w5, w10, errs = simulate(mw, xw, dw, cal, bw)
corr_raw, corr_true = build_curve(cal, bw)

print(f"\nBest curve points:")
print(f"  RAW:  {corr_raw.tolist()}")
print(f"  TRUE: {corr_true.tolist()}")

print(f"\nPer-sample errors (best config):")
for s, e in zip(labeled, errs):
    w = compute_weighted(s['ages'], mw, xw, dw)
    c = float(np.interp(w, corr_raw, corr_true))
    mark = "OK" if e <= 5 else "FAIR" if e <= 10 else "MISS"
    print(f"  [{mark:4s}] GT:{s['gt']:3d} Pred:{int(round(c)):3d} Err:{e:3d} "
          f"Weighted:{w:.1f}  {s['name']}")

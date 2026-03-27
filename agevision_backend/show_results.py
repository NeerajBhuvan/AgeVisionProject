import json

data = json.load(open('media/test_images/accuracy_report.json'))
m = data['metrics']
print(f"MAE: {m['mae']:.2f}")
print(f"Median Error: {m['median_ae']:.1f}")
print(f"Within +-3yr: {m['within_3_pct']:.1f}%")
print(f"Within +-5yr: {m['within_5_pct']:.1f}%")
print(f"Within +-10yr: {m['within_10_pct']:.1f}%")
print(f"RMSE: {m['rmse']:.2f}")
print()
print("Per age group:")
for g, v in m['per_age_group'].items():
    print(f"  {g:6s}: MAE={v['mae']:.1f} (n={v['count']})")
print()
print("Per-sample results:")
for r in data['results']:
    gt = r.get('ground_truth_age')
    if not gt:
        continue
    e = abs(r['predicted_age'] - gt)
    mark = "OK" if e <= 5 else "FAIR" if e <= 10 else "MISS"
    print(f"  [{mark:4s}] {r['description']:35s} GT:{gt:3d} Pred:{r['predicted_age']:3d} Err:{e:3d}")

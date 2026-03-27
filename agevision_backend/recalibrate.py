"""
Re-calibrate the correction curve using the new 7-detector ensemble data.
Reads the latest accuracy_report.json and computes optimal calibration points.
"""
import json
import numpy as np

data = json.load(open('media/test_images/accuracy_report.json'))

print("=== Re-calibrating correction curve from 7-detector data ===\n")

cal_points = []
for r in data['results']:
    gt = r.get('ground_truth_age')
    if not gt:
        continue
    ages = [float(a) for a in r['ensemble_ages']]
    a = np.array(ages)
    mean_age = float(np.mean(a))
    max_age = float(np.max(a))
    med_age = float(np.median(a))
    weighted = 0.50 * mean_age + 0.30 * max_age + 0.20 * med_age
    err = r['predicted_age'] - gt
    cal_points.append((weighted, gt, r['description'], err))
    print(f"  Weighted:{weighted:5.1f} → GT:{gt:3d}  (Pred:{r['predicted_age']:3d} Err:{err:+3d})  {r['description']}")

cal_points.sort(key=lambda x: x[0])

print(f"\n--- Sorted calibration points (weighted → true) ---")
print("_CALIBRATION_POINTS = np.array([")
for w, gt, desc, err in cal_points:
    print(f"    [{w:5.1f}, {gt:3d}],   # {desc}")
print("], dtype=float)")

# Also try different max weights
print("\n--- Simulating different max-weights ---")
for max_w in [0.10, 0.15, 0.20, 0.25, 0.30]:
    mean_w = 1.0 - max_w - 0.20
    errors = []
    for r in data['results']:
        gt = r.get('ground_truth_age')
        if not gt:
            continue
        ages = [float(a) for a in r['ensemble_ages']]
        a = np.array(ages)
        weighted = mean_w * np.mean(a) + max_w * np.max(a) + 0.20 * np.median(a)
        errors.append(abs(weighted - gt))
    mae = np.mean(errors)
    print(f"  max_w={max_w:.2f} mean_w={mean_w:.2f}: raw_MAE={mae:.1f}")

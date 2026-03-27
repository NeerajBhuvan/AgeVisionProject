import json

d = json.load(open('media/test_images/accuracy_report.json'))
for r in d['results']:
    gt = r.get('ground_truth_age')
    if gt is not None:
        ea = r.get('ensemble_ages', [])
        desc = r.get('description', '?')[:32]
        race = r.get('race', '?')
        pred = r['predicted_age']
        std = r.get('age_std', 0)
        raw_mean = sum(ea) / len(ea) if ea else 0
        print(f"GT:{gt:>3} Pred:{pred:>3} RawMean:{raw_mean:>5.1f} Ensemble:{ea} race={race} std={std:.1f} | {desc}")

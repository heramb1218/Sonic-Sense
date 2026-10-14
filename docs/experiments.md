# Experiments Log

This document tracks all experiments conducted during the project.

---

## Experiment Template

### Experiment N: [Title]
- **Date**: YYYY-MM-DD
- **Objective**: What are we testing?
- **Setup**: Parameters, data split, hardware
- **Results**: Metrics, plots
- **Observations**: What did we learn?
- **Limitations**: What could affect validity?

---

## Planned Experiments

1. **CNN Baseline** — Train CNN on UrbanSound8K with Log-Mel features
2. **CNN + Augmentation** — Compare with/without data augmentation
3. **CNN vs Transformer** — Architecture comparison
4. **Calibration Effect** — Uncalibrated vs calibrated sensor readings
5. **H3 Resolution** — Effect of spatial resolution on aggregation quality
6. **Anomaly Detection** — Statistical baseline vs Isolation Forest
7. **Spatial Resolution** — Raw points vs H3 aggregated view

---

*Results will be added as experiments are completed.*


### Experiment 1: CNN Baseline Evaluation
- **Checkpoint**: `cnn_urban_sound.pth`
- **Test Samples**: 100
- **Accuracy**: 20.00%
- **Macro Precision**: 4.76%
- **Macro Recall**: 20.00%
- **Macro F1-Score**: 7.50%
- **Weighted F1-Score**: 7.50%
- **Artifacts**:
  - Confusion Matrix: `results/confusion_matrix.png`
  - Full Report: `results/evaluation_report.json`

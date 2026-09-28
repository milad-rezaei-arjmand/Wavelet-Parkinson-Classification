# Results

This directory contains selected final outputs and visualizations from the wavelet-based Parkinson's disease classification study.

The complete set of intermediate experiment artifacts is not tracked in Git. Generated working files are written to the local `outputs/` directory.

---

## Structure

```text
results/
├── figures/
│   ├── project_pipeline.png
│   ├── model_accuracy_progression.png
│   ├── swt_consensus_confusion_matrix.png
│   ├── swt_fivefold_confusion_matrix.png
│   ├── swt_repeated_accuracy.png
│   ├── wpt_accuracy_coverage_curve.png
│   ├── wpt_risk_coverage_curve.png
│   └── wpt_vs_swt_comparison.png
├── metrics/
│   ├── swt_final_summary.csv
│   ├── swt_fold_results.csv
│   ├── swt_repeat_results.csv
│   ├── wavelet_representation_comparison.csv
│   ├── wpt_final_summary.csv
│   └── wpt_repeat_results.csv
└── reports/
    ├── FINAL_RESULTS.txt
    └── consensus_classification_report.txt
```

---

## Final SWT Evaluation

Repeated nested subject-level validation:

- Accuracy: **87.72% ± 0.25%**
- Balanced Accuracy: **84.96%**
- Macro-F1: **83.13%**
- ROC-AUC: **92.68%**

Consensus across five repetitions:

- Accuracy: **89.01%**
- Balanced Accuracy: **86.16%**
- Macro-F1: **84.73%**
- ROC-AUC: **93.37%**

---

## Validation Note

All reported final results use subject-level grouping to prevent windows from the same participant from appearing in both training and testing partitions.

The WPT selective-classification analysis is exploratory. Accuracy values obtained after referring low-confidence subjects are conditional on reduced coverage and must not be interpreted as full-coverage model accuracy.

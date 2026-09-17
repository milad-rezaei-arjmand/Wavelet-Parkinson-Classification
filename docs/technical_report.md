# Wavelet-Based Parkinson's Disease Classification Using Inertial Wrist Signals

## Project Overview

This project develops a complete machine learning pipeline for
distinguishing healthy subjects from Parkinson's disease patients using
inertial wrist sensor signals, wavelet-based feature extraction, and
support vector machine classification.

The main focus is a leakage-free subject-level evaluation framework
comparing multiple wavelet representations:

-   Discrete Wavelet Transform (DWT)
-   Wavelet Packet Transform (WPT)
-   Stationary Wavelet Transform (SWT)

The final selected approach uses SWT features combined with an RBF-SVM
classifier.

------------------------------------------------------------------------

# Dataset Description

The original dataset contained:

-   469 MATLAB files
-   Wrist inertial recordings from left and right wrists
-   11 motor activities
-   6 channels per wrist:
    -   3 accelerometer channels
    -   3 gyroscope channels
-   Sampling frequency: 100 Hz

After diagnosis filtering:

  Class         Subjects
  ----------- ----------
  Parkinson          276
  Healthy             79
  Total              355

Subjects with non-target diagnoses were excluded.

------------------------------------------------------------------------

# Processing Pipeline

    MATLAB Signal Files
            |
            v
    Data Inspection and Filtering
            |
            v
    Signal Windowing
            |
            v
    Wavelet Feature Extraction
    (DWT / WPT / SWT)
            |
            v
    Feature Selection
            |
            v
    Machine Learning Models
    (SVM / Random Forest / Ensembles)
            |
            v
    Subject-Level Nested Validation
            |
            v
    Final Evaluation

------------------------------------------------------------------------

# Signal Preprocessing

Signals were divided into fixed windows:

-   Window length: 512 samples
-   Duration: 5.12 seconds
-   Overlap: 50%
-   Step size: 256 samples

This produced:

-   45 windows per subject
-   15,975 total windows

A subject-level splitting strategy was used to prevent data leakage. All
windows from the same subject remained exclusively in either training or
testing partitions.

------------------------------------------------------------------------

# Wavelet Feature Extraction

Statistical features were extracted from wavelet coefficients:

-   Mean
-   Variance
-   Standard deviation
-   Energy
-   Relative energy
-   Shannon entropy
-   Interquartile range
-   Skewness
-   Kurtosis

Wavelet configurations:

  Method   Configuration     Features
  -------- --------------- ----------
  DWT      db4, level 5           648
  WPT      db4, level 4          1728
  SWT      db4, level 5           648

------------------------------------------------------------------------

# Machine Learning Framework

Models evaluated:

-   RBF Support Vector Machine
-   Linear SVM
-   Random Forest
-   Activity-based ensembles
-   Stacking models
-   Feature fusion approaches

Evaluation strategy:

-   Stratified Group K-Fold
-   Nested Cross Validation
-   Subject-based grouping
-   Repeated validation

Evaluation metrics:

-   Accuracy
-   Balanced Accuracy
-   Macro F1-score
-   Precision
-   Recall
-   ROC-AUC
-   Confusion Matrix

------------------------------------------------------------------------

# Final Model

The final selected model:

    Stationary Wavelet Transform (SWT)

    Wavelet:
    db4

    Level:
    5

    Features:
    648 → 300 selected features

    Classifier:
    RBF-SVM

    Parameters:
    C = 10
    gamma = scale
    class_weight = balanced

------------------------------------------------------------------------

# Final Results

Repeated SWT validation:

  Metric                        Result
  ------------------- ----------------
  Accuracy              87.72% ± 0.25%
  Balanced Accuracy             84.96%
  Macro F1                      83.13%
  ROC-AUC                       92.68%

Consensus across five independent repetitions:

  Metric       Result
  ---------- --------
  Accuracy     89.01%
  ROC-AUC      93.37%

------------------------------------------------------------------------

# Key Findings

-   SWT provided the most stable overall representation.
-   Removing downsampling in SWT helped preserve temporal patterns.
-   Subject-level validation prevented overly optimistic results caused
    by window leakage.
-   Fusion methods did not consistently improve generalization.
-   Confidence-based selective analysis showed higher accuracy when
    uncertain cases were referred.

------------------------------------------------------------------------

# Repository Structure

    Wavelet-Parkinson-Classification/

    ├── src/
    │   ├── preprocessing/
    │   ├── wavelet_features/
    │   ├── models/
    │   └── analysis/
    │
    ├── results/
    │   ├── figures/
    │   ├── metrics/
    │   └── reports/
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Reproducibility

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run preprocessing, feature extraction, training, and evaluation scripts
from the `src` directory.

------------------------------------------------------------------------

# Conclusion

This project presents a complete wavelet-based machine learning
framework for Parkinson's disease classification from wrist inertial
signals.

The main contribution is the comparison of multiple wavelet
representations under a strict subject-level validation protocol,
demonstrating the importance of leakage prevention and robust evaluation
in biomedical signal classification.

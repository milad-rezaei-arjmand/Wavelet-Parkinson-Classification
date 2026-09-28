# Dataset

## Data Availability

The raw Parkinson's disease inertial-sensor dataset used in this project is **not distributed through this repository**.

Users must obtain the dataset separately and comply with the original provider's access, licensing, and usage requirements.

---

## Expected Local Layout

After obtaining the data, place the MATLAB files in:

```text
data/
└── raw/
    ├── 001.mat
    ├── 002.mat
    ├── 003.mat
    └── ...
```

The `data/raw/` directory is intentionally excluded from version control.

---

## Dataset Characteristics Used by the Pipeline

The project expects wrist inertial recordings with:

- left and right wrist signals,
- 3 accelerometer channels per wrist,
- 3 gyroscope channels per wrist,
- multiple motor activities,
- and subject-level diagnosis labels.

The experiments use a sampling frequency of 100 Hz.

The original dataset contained 469 MATLAB files. After diagnosis filtering, the analysis retained 355 subjects:

- 276 Parkinson subjects
- 79 healthy controls

Subjects with non-target diagnoses were excluded.

---

## Preprocessing

The preprocessing scripts:

```text
src/preprocessing/01_inspect_files.py
src/preprocessing/02_build_inventory.py
src/preprocessing/03_build_window_inventory.py
```

inspect the MATLAB files, standardize target labels, validate signal tables, and construct the window inventory used by downstream wavelet feature-extraction scripts.

Windows are generated using:

- window length: 512 samples
- overlap: 50%
- step size: 256 samples

Raw dataset files must not be committed to this repository.

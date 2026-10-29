# Data: WM-811K

Wafer map defect-pattern dataset from a real fab, publicly released for research (distributed via Kaggle as "WM-811K (Wafer Map)"; underlying file `LSWMD.pkl`).

## How to obtain

1. Download `LSWMD.pkl` from the Kaggle dataset page (respect the licence/terms stated there).
2. Place it at `data/raw/LSWMD.pkl` (the `data/raw/` directory is git-ignored; raw data is never committed).

## Expected schema (columns of the pickled DataFrame)

| Column | Meaning |
|---|---|
| `dieSize` | Die size of the wafer map |
| `lotName` | Lot identifier — used for grouped train/test splits |
| `waferIndex` | Wafer index within the lot |
| `waferMap` | 2D array of die states (0 = no die, 1 = good, 2 = defective) |
| `trainTestLabel` | Original train/test marker from the release |
| `failureType` | Defect-pattern label (subset of rows only) |

Roughly 811k maps in total; the labelled subset is ~25k rows across 9 classes (`none` plus 8 defect patterns), heavily imbalanced toward `none`. Exact counts are fixed by the EDA notebook and recorded in the main README.

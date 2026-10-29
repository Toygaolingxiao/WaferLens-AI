# WaferLens-AI

Wafer map defect-pattern classification with measured method comparison — statistics and models decide, an AI agent explains, every number fact-checked.

> Status: project scaffold. Models, evaluation tables and the explanation agent land step by step per the roadmap below; until then, tables marked TBD are placeholders, not results.

## What this is

A personal project applying one discipline to semiconductor yield analytics: several modelling routes run on the **same data, same split, same metrics**, the numbers land in `results/` before any story is told, and an LLM agent may only explain what the statistics computed — every figure it cites is checked against the computed values first.

Dataset: **WM-811K** wafer maps (real fab data, public release): ~811k maps, a labelled subset of ~25k across 9 classes (`none` plus Center / Donut / Edge-Loc / Edge-Ring / Loc / Random / Scratch / Near-Full). See `data/README.md`.

## Method comparison (results land in `results/`)

| Route | Method | Macro-F1 | Notes |
|---|---|---|---|
| A | Hand-crafted geometry/density features + RandomForest / XGBoost | TBD | Interpretable baseline |
| B | Small CNN (PyTorch) on raw wafer maps | TBD | Mean ± spread over ≥2 seeds |
| C | Pre-trained visual features + linear head (optional) | TBD | Cut first if time runs short |

Primary metric: macro-F1. Secondary: per-class recall (minority classes such as Scratch and Near-Full reported explicitly), confusion matrix.

## Architecture

```
L1 data      loader, cleaning, two-stage labels, lot-grouped split (no leakage)
L2 models    routes A / B / C in parallel
L3 eval      the single comparison table -> results/ (only source of numbers)
L4 spc       defect-rate p-chart + EWMA on daily rates (standalone module)
L5 agent     explanation agent over L3/L4 outputs, fact-checked numbers
```

Modelling is **two-stage**: stage 1 separates defective from `none`; stage 2 classifies the defect pattern. Splits are grouped by lot so wafers from one lot never straddle train/test.

## Evaluation protocol (fixed in advance)

- Metrics decided before modelling; no post-hoc metric shopping.
- All routes share one split and one metric set.
- Explanation agent: 10 fixed cases, human-scored for process plausibility, plus a numeric fact-check pass rate.
- Failures are reported in the same table as wins.

## Repository layout

- `src/waferlens/` — data / features / models / spc / agent / eval packages
- `results/` — comparison tables, figures, agent case scores (committed)
- `notebooks/` — EDA
- `docs/design-notes.md` — decision log (why two-stage, why lot-grouped splits, cut order)

## Limitations, stated plainly

Public benchmark data from one fab's release — not a production line, not this author's employer's data. This project demonstrates method and evaluation hygiene; it does not claim production results.

## Roadmap

- Weekend 1: L1 + EDA, routes A/B + comparison table v1, SPC charts
- Weekend 2: L5 explanation agent + fact-check, capped LoRA exercise (equipment-log text classification, LoRA vs prompt-only), README finalised with measured numbers

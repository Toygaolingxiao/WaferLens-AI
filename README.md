# WaferLens-AI

Wafer map defect-pattern classification with measured method comparison — statistics and models decide, an AI agent explains, every number fact-checked.

> Status: scaffold with the data layer and evaluation plumbing in place. Modelling routes, SPC charts and the explanation agent land per the roadmap below; tables marked TBD are placeholders, not results.

## Why wafer maps

When yield drops, the wafer map is often the first artefact an engineer looks at: defective dies are rarely scattered at random — they form spatial patterns, and each pattern is a fingerprint of a process step. Reading those patterns quickly and consistently is a core yield-engineering skill, and a natural place to test whether modern ML (and an LLM on top) actually helps or just adds noise. This project measures that, on public data, with the failure cases left in.

## Defect patterns and their typical process associations

Working hypotheses used by the explanation layer — associations to verify against process data, not ground truth:

| Pattern | Typical association |
|---|---|
| Center | Centre non-uniformity in deposition / spin processes |
| Donut | Radial non-uniformity: etch or plasma distribution, CMP |
| Edge-Ring | Edge etch / bevel effects, focus-ring wear, edge exclusion drift |
| Edge-Loc | Edge handling damage, notch-area defects, lithography edge issues |
| Loc | Local contamination or a localised process excursion |
| Random | Random particles / background contamination |
| Scratch | Handling or CMP scratches (transfer arms, polishing) |
| Near-Full | Catastrophic excursion — most of the wafer affected |

## What this project does

Several modelling routes run on the **same data, same split, same metrics**; numbers land in `results/` before any story is told; an LLM agent may only explain what the statistics computed, and every figure it cites is checked against the computed values first (see `src/waferlens/agent/factcheck.py`).

Dataset: **WM-811K** (real fab wafer maps, public release): ~811k maps, a labelled subset of ~25k across 9 classes (`none` plus the 8 patterns above), heavily imbalanced toward `none`. See `data/README.md` for acquisition and schema.

## Method comparison (results land in `results/`)

| Route | Method | Macro-F1 | Notes |
|---|---|---|---|
| A | Hand-crafted geometry/density features + RandomForest / XGBoost | TBD | Interpretable baseline |
| B | Small CNN (PyTorch) on raw wafer maps | TBD | Mean ± spread over ≥2 seeds |
| C | Pre-trained visual features + linear head (optional) | TBD | Cut first if time runs short |

Primary metric: macro-F1. Secondary: per-class recall (minority classes such as Scratch and Near-Full reported explicitly), confusion matrix.

Modelling is **two-stage**: stage 1 separates defective from `none`; stage 2 classifies the pattern among defective wafers. Imbalance handling (class weights vs resampling) is itself compared, not assumed.

## Architecture

```
L1 data      loader, cleaning, two-stage labels, lot-grouped split (no leakage)
L2 models    routes A / B / C in parallel
L3 eval      the single comparison table -> results/ (only source of numbers)
L4 spc       defect-rate p-chart + EWMA on daily rates (standalone module)
L5 agent     explanation agent over L3/L4 outputs, fact-checked numbers
```

Splits are grouped by lot (`lotName`): wafers from one lot never straddle train/test, because same-lot wafers share process conditions and a row-wise split would leak them.

## Quickstart

```bash
pip install -e ".[dev]"
# download LSWMD.pkl (see data/README.md) into data/raw/
python -m pytest            # split-hygiene and fact-check tests
```

## Evaluation protocol (fixed in advance)

- Metrics decided before modelling; no post-hoc metric shopping.
- All routes share one split and one metric set.
- Explanation agent: 10 fixed cases, human-scored for process plausibility, plus a numeric fact-check pass rate.
- Failures are reported in the same table as wins.

## Repository layout

- `src/waferlens/` — data / features / models / spc / agent / eval packages
- `results/` — comparison tables, figures, agent case scores (committed)
- `notebooks/` — EDA (`01_eda.ipynb` fixes the exact class counts quoted here)
- `docs/design-notes.md` — decision log (why two-stage, why lot-grouped splits, cut order)

## Limitations, stated plainly

Public benchmark data from one fab's release — not a production line, and not any employer's data. Labelled maps are a small, imbalanced subset of the release. This project demonstrates method and evaluation hygiene; it does not claim production results.

## Roadmap

- Weekend 1: EDA + exact counts, routes A/B + comparison table v1, SPC charts
- Weekend 2: L5 explanation agent, capped LoRA exercise (equipment-log text classification, LoRA vs prompt-only), README finalised with measured numbers

## References

- Wu, Jen & Deng (2015), wafer map failure pattern recognition work behind the WM-811K release (MIR Lab). Dataset distributed via Kaggle as "WM-811K (Wafer Map)".
- Methodological sibling: [ResiliMES-AI](https://github.com/Toygaolingxiao/ResiliMES-AI) — same discipline (deterministic core decides, AI explains, numbers fact-checked) applied to fab scheduling.

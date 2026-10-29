# Design notes (decision log)

## Why two-stage modelling

The labelled subset is dominated by `none`. A single 9-class model spends most
of its capacity on the majority class and hides minority-pattern performance.
Stage 1 (defective vs none) and stage 2 (pattern within defective) keep both
problems visible and separately measurable.

## Why lot-grouped splits

Wafers from the same lot share process conditions; a random row-wise split
leaks lot identity into both sides and inflates scores. Splits are grouped by
`lotName`. This is the project's hygiene signature: slightly worse numbers,
honestly earned.

## Metrics fixed in advance

Primary metric macro-F1; per-class recall and the confusion matrix are always
reported alongside. The protocol was written before any model was trained;
changing it later requires a note in this file explaining why.

## The agent explains; it does not decide

Classification and SPC state come from code. The LLM layer only verbalises
them (pattern explanation, candidate process causes, next checks), and every
number it cites passes a code fact-check against the computed values first.
A failed check rewrites or refuses — the same guardrail pattern as the
author's scheduling work.

## Cut order (if time runs short)

Route C → LoRA exercise → SPC. Protected to the end: L1 hygiene and the
route A/B comparison table.

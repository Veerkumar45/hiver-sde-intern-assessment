# Hiver SDE Intern — Customer Support AI Agent

This repository implements the Hiver take-home around a brand-specific customer-support agent. The assignment requires a runnable pipeline, a 150–250 example hand-labelled golden set, automated evaluation plus an LLM judge, baselines, failure analysis, a misleading-headline-number section, and a decision log.

## Current status

The repository is intentionally split into:
- a **smoke/demo path** that runs immediately using the included sample;
- a **submission evaluation path** that requires genuine human labels.

No golden labels are fabricated.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
```

## 1. Smoke test

```powershell
python -m src.pipeline --data data/raw/sample.csv --brand AppleSupport --limit 100
python -m src.evaluate --predictions outputs/predictions.csv
```

## 2. Real dataset

Put the official Kaggle dataset at:

```text
data/raw/twcs.csv
```

Then run:

```powershell
python -m src.pipeline --data data/raw/twcs.csv --brand AppleSupport --limit 50000
```

This creates `outputs/historical_pairs.csv`.

## 3. Create the 200-example golden set

```powershell
python -m src.create_golden_set --pairs outputs/historical_pairs.csv --n 200 --out data/golden/golden_eval.csv
```

Open:

```text
data/golden/golden_eval.csv
```

Read `data/golden/LABELING_GUIDE.md`.

Fill the **intent** column with human labels. Do not use the model's predictions as labels.

The assignment requires 150–250 hand-labelled examples, so 200 is inside the requested range.

## 4. Run leakage-safe golden evaluation

```powershell
python -m src.pipeline --data data/raw/twcs.csv --brand AppleSupport --limit 50000 --golden data/golden/golden_eval.csv
```

This creates:
- `outputs/golden_predictions.csv`
- `outputs/metrics.json`

The golden examples are excluded from the training/retrieval corpus by text, and retrieval explicitly excludes the evaluated tweet ID. This avoids the previous self-retrieval problem.

## 5. Compare baselines

```powershell
python -m src.evaluate --predictions outputs/golden_predictions.csv --golden data/golden/golden_eval.csv
```

This reports:
- majority baseline accuracy / macro F1
- keyword baseline accuracy / macro F1
- retrieval-agent accuracy / macro F1
- per-intent precision / recall / F1

## 6. LLM-as-judge

Set an API key in your terminal:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

Then:

```powershell
python -m src.evaluate --predictions outputs/golden_predictions.csv --golden data/golden/golden_eval.csv --judge-sample 30
```

The judge scores:
- groundedness
- relevance
- safety
- concision

For the final assessment, have a human independently rate the same judge sample and report agreement. The repository does not invent that agreement.

## Architecture

```text
Twitter CSV
   ↓
conversation reconstruction
   ↓
AppleSupport pairs
   ↓
intent classifier
   ↓
historical retrieval (self-match excluded)
   ↓
reply draft
   ↓
escalation gate
   ↓
evaluation
```

## Intent taxonomy

- `update_performance_or_crash`
- `battery_drain`
- `app_or_media_behavior`
- `account_or_verification`
- `device_issue_other`

## Safety

The agent escalates low-confidence, weak-evidence, sensitive, or high-risk cases. It does not autonomously change accounts, process refunds, or request passwords.

## Deliverables

- `README.md` — reproduction instructions
- `data/golden/golden_eval.csv` — after human labelling
- `data/golden/LABELING_GUIDE.md`
- `reports/assessment_report.md`
- `DECISION_LOG.md`
- `src/` — runnable implementation
- `tests/` — tests

## Important

The included sample is for smoke testing. Final claims should be made only after the 200-example golden set has been genuinely hand-labelled and frozen.
